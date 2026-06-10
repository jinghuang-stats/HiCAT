import os
import pickle 
import warnings
warnings.filterwarnings('ignore')
import time
import statistics
import hnswlib
import math
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import anndata as ad
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from anndata import AnnData
from scipy.sparse import issparse
from scipy.stats import rankdata
from scipy.spatial.distance import cdist, pdist, squareform


def _unique_preserve_order(items):
	seen=set()
	ordered=[]
	for item in items:
		if item not in seen:
			seen.add(item)
			ordered.append(item)
	return ordered


def _validate_obs_columns(adata, columns, context="adata"):
	missing=[col for col in columns if col not in adata.obs.columns]
	if len(missing)>0:
		raise KeyError(f"{context} is missing required obs columns: {missing}")


def _safe_value_counts(series, categories=None):
	counts=series.value_counts()
	if categories is not None:
		counts=counts.reindex(categories, fill_value=0)
	return counts


def _safe_quantile(values, q, default=0):
	if hasattr(values, "toarray"):
		arr=values.toarray()
	else:
		arr=np.asarray(values)
	arr=np.asarray(arr).ravel()
	arr=arr[np.isfinite(arr)]
	if arr.size==0:
		return default
	return float(np.quantile(arr, q))


def _safe_cluster_ids_from_index(index_like, prefix="cluster_"):
	ids=[]
	for item in index_like:
		try:
			ids.append(int(str(item).replace(prefix, "")))
		except Exception:
			pass
	return ids


def _category_codes(series):
	return pd.Categorical(series).codes


def _cluster_with_scanpy(features_matrix, method, resolution, n_neighbors, n_clusters=None, random_state=0, cluster_key="cluster", approx_pca=None):
	if hasattr(features_matrix, "toarray"):
		x=features_matrix.toarray()
	else:
		x=np.asarray(features_matrix)
	tmp=sc.AnnData(x)
	if x.shape[0] == 0:
		raise ValueError("Cannot cluster an empty feature matrix")
	if method in ["leiden", "louvain"]:
		sc.pp.neighbors(tmp, n_neighbors=n_neighbors, random_state=random_state)
		if method=="leiden":
			sc.tl.leiden(tmp, resolution=resolution, key_added=cluster_key, random_state=random_state)
		else:
			sc.tl.louvain(tmp, resolution=resolution, key_added=cluster_key, random_state=random_state)
		y_pred=_category_codes(tmp.obs[cluster_key])
	elif method=="kmeans":
		if n_clusters is None:
			raise ValueError("n_clusters is required when method='kmeans'")
		kmeans=KMeans(n_clusters=n_clusters, random_state=random_state)
		y_pred=kmeans.fit_predict(x)
	else:
		raise ValueError(f"Unsupported clustering method: {method}")
	return y_pred


def _cluster_spatial_pcs(adata, n_components, method, resolution, n_neighbors, cluster_key, nodes_num, random_state=0, large_clusters=True, n_clusters=None):
	if n_components is None or n_components <= 0:
		raise ValueError("n_components must be a positive integer")
	pca=PCA(n_components=n_components, random_state=random_state)
	pcs=pca.fit_transform(adata.X)
	pcs=pcs-np.min(pcs)
	print("The value range for pcs: "+str(np.min(pcs))+" ~ "+str(np.max(pcs)))
	pred=_cluster_with_scanpy(features_matrix=pcs, method=method, resolution=resolution, n_neighbors=n_neighbors, n_clusters=n_clusters, random_state=random_state, cluster_key=cluster_key)
	if method=="leiden" and large_clusters is True:
		cluster_num=len(pd.Series(pred).value_counts())
		if cluster_num>(int(2*nodes_num)+1):
			print("--------- !!! Leiden resolution tends to be too large !!! ---------")
			pred=_cluster_with_scanpy(features_matrix=pcs, method=method, resolution=np.max([resolution-0.2, 0.1]), n_neighbors=n_neighbors, random_state=random_state, cluster_key=cluster_key)
	if method=="leiden" and len(pd.Series(pred).value_counts())<=2:
		print("--------- !!! Leiden resolution tends to be too small !!! ---------")
		pred=_cluster_with_scanpy(features_matrix=pcs, method=method, resolution=np.min([resolution+0.2,0.8]), n_neighbors=n_neighbors, random_state=random_state, cluster_key=cluster_key)
	adata.obs[cluster_key]=pred.copy()
	if method=="kmeans":
		adata.obs[cluster_key]=adata.obs[cluster_key].astype("category")
	return adata


def kmeans_clustering(features_matrix, n_clusters=5, random_state=0, kmeans_key="kmeans_clusters"):
	y_pred=_cluster_with_scanpy(features_matrix=features_matrix, method="kmeans", resolution=None, n_neighbors=None, n_clusters=n_clusters, random_state=random_state, cluster_key=kmeans_key)
	print("========== KMeans Clustering Results ==========")
	print(pd.Series(y_pred).value_counts())
	return y_pred


# testified -> used to check the utility of HIPT PCs
def leiden_clustering(features_matrix, resolution, n_neighbors, random_state=0, leiden_key="leiden_clusters"):
	y_pred=_cluster_with_scanpy(features_matrix=features_matrix, method="leiden", resolution=resolution, n_neighbors=n_neighbors, random_state=random_state, cluster_key=leiden_key)
	print("========== Leiden Clustering Results ==========")
	print(pd.Series(y_pred).value_counts())
	return y_pred


# upd version of louvain_clustering
def louvain_clustering(features_matrix, resolution=0.1, n_neighbors=10, random_state=0, louvain_key="louvain_clusters"):
	y_pred=_cluster_with_scanpy(features_matrix=features_matrix, method="louvain", resolution=resolution, n_neighbors=n_neighbors, random_state=random_state, cluster_key=louvain_key)
	print("========== Louvain Clustering Results ==========")
	print(pd.Series(y_pred).value_counts())
	return y_pred


# label refinement function
def refine_labels(input_adata, pred_key, refined_key, num_nbs, x_col="x", y_col="y", dists_metric="euclidean"):
	# shape = "square" for ST data | "hexagon" for Visium data
	refined_pred=[]
	spot_id=input_adata.obs.index.tolist()
	pred=input_adata.obs[pred_key].copy()
	pred=pd.DataFrame({"pred": pred}, index=spot_id)
	# calculate pairwise distances
	spatial_coords=input_adata.obs[[x_col,y_col]]
	spatial_df=pd.DataFrame(spatial_coords, index=spot_id)
	dists=squareform(pdist(spatial_df, metric="euclidean"))
	dists=pd.DataFrame(dists, index=spot_id, columns=spot_id)
	# refine prediction results
	for i in range(len(spot_id)):
		index=spot_id[i]
		dists_tmp=dists.loc[index, :].sort_values()
		nbs=dists_tmp[0:num_nbs+1]
		nbs_pred=pred.loc[nbs.index, "pred"]
		self_pred=pred.loc[index, "pred"]
		v_c=nbs_pred.value_counts()
		if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
			refined_pred.append(v_c.idxmax())
		else:
			refined_pred.append(self_pred)
	input_adata.obs[refined_key]=refined_pred.copy()
	input_adata.obs[refined_key]=input_adata.obs[refined_key].astype("category")
	return input_adata, refined_pred


def _hipt_pcs_clustering_impl(hipt_adata, hipt_npcs, resolution, n_neighbors, cluster_key, nodes_num, random_state=0, large_clusters=True, method="leiden", n_clusters=None):
	hipt_adata=_cluster_spatial_pcs(adata=hipt_adata, n_components=hipt_npcs, method=method, resolution=resolution, n_neighbors=n_neighbors, cluster_key=cluster_key, nodes_num=nodes_num, random_state=random_state, large_clusters=large_clusters, n_clusters=n_clusters)
	print("The final number of hipt clusters is "+str(len(pd.Series(hipt_adata.obs[cluster_key]).value_counts())))
	return hipt_adata


def _identify_boundary_cluster_impl(hipt_adata, cluster_key):
	hipt_clusters=hipt_adata.obs[cluster_key].value_counts().index.tolist()
	target_features_1=["hipt_576", "hipt_577", "hipt_578"]
	target_features_2=["rgb_0","rgb_1","rgb_2"]
	is_subset_1=all(f1 in hipt_adata.var.index.tolist() for f1 in target_features_1)
	is_subset_2=all(f2 in hipt_adata.var.index.tolist() for f2 in target_features_2)
	if is_subset_1:
		rgb_features=target_features_1
	elif is_subset_2:
		rgb_features=target_features_2
	else:
		raise KeyError("identify_bd_cluster requires either ['hipt_576','hipt_577','hipt_578'] or ['rgb_0','rgb_1','rgb_2'] in hipt_adata.var.index")
	mean_rgb_df=pd.DataFrame(np.zeros((len(hipt_clusters), len(rgb_features))), index=hipt_clusters, columns=rgb_features)
	for k in hipt_clusters:
		cluster_mask=hipt_adata.obs[cluster_key]==k
		cluster_rgb_tmp=hipt_adata[cluster_mask, hipt_adata.var.index.isin(rgb_features)].copy()
		mean_rgb=np.asarray(cluster_rgb_tmp.X.mean(axis=0)).ravel()
		mean_rgb_df.loc[k,:]=mean_rgb
	print("========================= mean HIPT rgb values =========================")
	print(mean_rgb_df)
	mean_rgb_df["rgb_sum"]=mean_rgb_df.abs().sum(axis=1)
	bd_cluster=mean_rgb_df["rgb_sum"].idxmin()
	print("The boundary cluster is "+str(bd_cluster))
	return bd_cluster


def identify_bd_cluster(hipt_adata, cluster_key):
	return _identify_boundary_cluster_impl(hipt_adata=hipt_adata, cluster_key=cluster_key)


def identify_bd_cluster_upd(hipt_adata, cluster_key):
	return _identify_boundary_cluster_impl(hipt_adata=hipt_adata, cluster_key=cluster_key)


def reassign_bd_cluster(hipt_adata, bd_cluster, cluster_key, x_col, y_col, bd_num_nbs_1=25, bd_num_nbs_2=15):
	#----------------------- 1. calculate the pairwise distances between the boundary cluster and its neighboring clusters -----------------------
	# boundary cluster
	bd_df=hipt_adata[hipt_adata.obs[cluster_key]==bd_cluster].obs.copy()
	bd_pred=pd.DataFrame({"pred": bd_df[cluster_key]}, index=bd_df.index.tolist())
	bd_coords=bd_df[[x_col,y_col]] # enhanced data ["x", "y"]
	# other clusters
	other_df=hipt_adata[hipt_adata.obs[cluster_key]!=bd_cluster].obs.copy()
	other_pred=pd.DataFrame({"pred": other_df[cluster_key]}, index=other_df.index.tolist())
	other_coords=other_df[[x_col, y_col]] # enhanced data ["x", "y"]
	# calculate pairwise distances
	dists=cdist(np.array(bd_coords), np.array(other_coords), metric="euclidean")
	dists=pd.DataFrame(dists, index=bd_df.index.tolist(), columns=other_df.index.tolist())
	#----------------------- 2. reassign boundary cluster identities -----------------------
	# the identity was reassigned to the neighboring cluster which takes the largest proportion in the neighborhood of each boundary spot
	refined_bd_key="refined_bd_clusters"
	refined_bd_pred=[]
	for i in range(dists.shape[0]):
		index=dists.index.tolist()[i]
		dists_tmp=dists.loc[index, :].sort_values()
		nbs=dists_tmp.iloc[0:min(bd_num_nbs_1, len(dists_tmp))]
		nbs_pred=other_pred.loc[nbs.index, "pred"]
		v_c=nbs_pred.value_counts()
		if len(v_c)==0:
			refined_bd_pred.append(bd_cluster)
		else:
			refined_bd_pred.append(v_c.idxmax())
	# refine the boundary cluster identities
	hipt_adata.obs[refined_bd_key]=hipt_adata.obs[cluster_key].copy()
	#hipt_adata.obs[refined_bd_key]=hipt_adata.obs[refined_bd_key].astype(int) # cannot change values for categorical values
	hipt_adata.obs.loc[bd_df.index.tolist(), refined_bd_key]=refined_bd_pred
	print("========================= refined boundary clusters =========================")
	print(hipt_adata.obs[refined_bd_key].value_counts())
	#----------------------- 3. refine HIPT clusters (overall) -----------------------
	# this step tends to be time-consuming (may skip it)
	final_cluster_key="final_"+cluster_key
	hipt_adata,_=refine_labels(input_adata=hipt_adata, pred_key=refined_bd_key, refined_key=final_cluster_key, num_nbs=bd_num_nbs_2, x_col=x_col, y_col=y_col)
	return hipt_adata, refined_bd_key, final_cluster_key


def subtypes_by_gene(hipt_adata, gene_adata, target_genes, nontgt_genes, subtype_gene_num, subtype_min_prop, subtype_num_nbs, x_col, y_col, final_cluster_key, resolution, n_neighbors, random_state=0):
	# use gene exp to identify subtypes within hipt clusters
	final_cluster_prop=hipt_adata.obs[final_cluster_key].value_counts(normalize=True)
	final_clusters=final_cluster_prop.index.tolist()
	print("========================= HIPT cluster prop =========================")
	print(final_cluster_prop)
	# select subset genes for subtyping
	subtype_genes=target_genes[0:subtype_gene_num]+nontgt_genes[0:subtype_gene_num]
	print("subtype genes used for louvain clustering: ")
	print("target genes: ["+", ".join(target_genes[0:subtype_gene_num])+"]")
	print("nontgt genes: ["+", ".join(nontgt_genes[0:subtype_gene_num])+"]")
	# identify subtype clusters
	subtype_key="subtype_clusters"
	hipt_adata.obs[subtype_key]=hipt_adata.obs[final_cluster_key].copy()
	hipt_adata.obs[subtype_key]=_category_codes(hipt_adata.obs[subtype_key]) # robust to string or categorical cluster labels
	for i in final_clusters:
		# no further splitting for too small clusters
		if final_cluster_prop[i]>=subtype_min_prop:
			print("------------------------ cluster "+str(i)+" louvain subtyping ------------------------")
			cluster_indices=hipt_adata[hipt_adata.obs[final_cluster_key]==i].obs.index.tolist()
			cluster_gene_tmp=gene_adata[gene_adata.obs.index.isin(cluster_indices), gene_adata.var.index.isin(subtype_genes)].copy()
			cluster_indices=cluster_gene_tmp.obs.index.tolist()
			# louvain clustering | kmeans clustering not work so well in this scenario
			subtype_pred=louvain_clustering(features_matrix=cluster_gene_tmp.X, resolution=resolution, n_neighbors=n_neighbors, random_state=random_state)
			# control the number of resulting clusters
			if len(pd.Series(subtype_pred).value_counts())>5:
				subtype_pred=louvain_clustering(features_matrix=cluster_gene_tmp.X, resolution=np.min([resolution,0.01]), n_neighbors=n_neighbors, random_state=random_state)
			# update the subtype column labeling
			hipt_adata.obs.loc[cluster_indices,subtype_key]=hipt_adata.obs.loc[cluster_indices,subtype_key].astype(str)+"_"+subtype_pred.astype(str)
	hipt_adata.obs[subtype_key]=hipt_adata.obs[subtype_key].astype("category")
	# update cluster labels: from *_* to integer values -> otherwise will occur errors in label assignment step
	upd_subtype_key="upd_"+subtype_key
	hipt_adata.obs[upd_subtype_key]=hipt_adata.obs[subtype_key].cat.codes
	print("========================= I + G subtyping clusters =========================")
	print(hipt_adata.obs[upd_subtype_key].value_counts())
	# refinement on subtype clusters (sparse patterns) | set num_nbs = 10 or 5 (may not be necessary)
	final_subtype_key="final_"+subtype_key
	# make the refinement as an optional step
	hipt_adata,_=refine_labels(input_adata=hipt_adata, pred_key=upd_subtype_key, refined_key=final_subtype_key, num_nbs=subtype_num_nbs, x_col=x_col, y_col=y_col)
	return hipt_adata, final_subtype_key


# removes the label refinement step
def subtypes_by_gene_v2(hipt_adata, gene_adata, target_genes, nontgt_genes, subtype_gene_num, subtype_min_prop, final_cluster_key, resolution, n_neighbors, random_state=0):
	# use gene exp to identify subtypes within hipt clusters
	final_cluster_prop=hipt_adata.obs[final_cluster_key].value_counts(normalize=True)
	final_clusters=final_cluster_prop.index.tolist()
	print("========================= HIPT cluster prop =========================")
	print(final_cluster_prop)
	# select subset genes for subtyping
	subtype_genes=target_genes[0:subtype_gene_num]+nontgt_genes[0:subtype_gene_num]
	print("subtype genes used for louvain clustering: ")
	print("target genes: ["+", ".join(target_genes[0:subtype_gene_num])+"]")
	print("nontgt genes: ["+", ".join(nontgt_genes[0:subtype_gene_num])+"]")
	# identify subtype clusters
	subtype_key="subtype_clusters"
	hipt_adata.obs[subtype_key]=hipt_adata.obs[final_cluster_key].copy()
	hipt_adata.obs[subtype_key]=_category_codes(hipt_adata.obs[subtype_key]) # robust to string or categorical cluster labels
	for i in final_clusters:
		# no further splitting for too small clusters
		if final_cluster_prop[i]>=subtype_min_prop:
			print("------------------------ cluster "+str(i)+" louvain subtyping ------------------------")
			cluster_indices=hipt_adata[hipt_adata.obs[final_cluster_key]==i].obs.index.tolist()
			cluster_gene_tmp=gene_adata[gene_adata.obs.index.isin(cluster_indices), gene_adata.var.index.isin(subtype_genes)].copy()
			cluster_indices=cluster_gene_tmp.obs.index.tolist()
			# louvain clustering | kmeans clustering not work so well in this scenario
			subtype_pred=louvain_clustering(features_matrix=cluster_gene_tmp.X, resolution=resolution, n_neighbors=n_neighbors, random_state=random_state)
			# control the number of resulting clusters
			if len(pd.Series(subtype_pred).value_counts())>5:
				subtype_pred=louvain_clustering(features_matrix=cluster_gene_tmp.X, resolution=np.min([resolution,0.01]), n_neighbors=n_neighbors, random_state=random_state)
			# update the subtype column labeling
			hipt_adata.obs.loc[cluster_indices,subtype_key]=hipt_adata.obs.loc[cluster_indices,subtype_key].astype(str)+"_"+subtype_pred.astype(str)
	hipt_adata.obs[subtype_key]=hipt_adata.obs[subtype_key].astype("category")
	# update cluster labels: from *_* to integer values -> otherwise will occur errors in label assignment step
	upd_subtype_key="upd_"+subtype_key
	hipt_adata.obs[upd_subtype_key]=hipt_adata.obs[subtype_key].cat.codes
	print("========================= I + G subtyping clusters =========================")
	print(hipt_adata.obs[upd_subtype_key].value_counts())
	return hipt_adata, upd_subtype_key


# G clustering function
# testified (based on gene exp features)
def gene_clustering(gene_adata, gene_features, resolution, n_neighbors, cluster_key, nodes_num, random_state=0, large_clusters=True):
	gene_exp_tmp=gene_adata[:,gene_adata.var.index.isin(gene_features)].X.copy()
	gene_pred=leiden_clustering(features_matrix=gene_exp_tmp, resolution=resolution, n_neighbors=n_neighbors, random_state=random_state, leiden_key=cluster_key)
	# if large_clusters is set as True -> will control the number of resulting clusters -> avoid too many small clusters
	#if large_clusters==True:
		#if len(pd.Series(gene_pred).value_counts())>(int(2*nodes_num)+1):
			#print("--------- !!! Leiden resolution tends to be too large !!! ---------")
			#gene_pred=leiden_clustering(features_matrix=gene_exp_tmp, resolution=np.min([resolution,0.15]), n_neighbors=n_neighbors, random_state=random_state, leiden_key=cluster_key)
	#if len(pd.Series(gene_pred).value_counts())<=2:
		#print("--------- !!! Leiden resolution tends to be too small !!! ---------")
		#gene_pred=leiden_clustering(features_matrix=gene_exp_tmp, resolution=np.max([resolution,0.5]), n_neighbors=n_neighbors, random_state=random_state, leiden_key=cluster_key)
	print("The final number of gene clusters is "+str(len(pd.Series(gene_pred).value_counts())))
	gene_adata.obs[cluster_key]=gene_pred.copy()
	return gene_adata


# based on kmeans clustering
def gene_clustering_kmeans(gene_adata, gene_features, n_clusters, cluster_key, random_state=0):
	gene_exp_tmp=gene_adata[:,gene_adata.var.index.isin(gene_features)].X.copy()
	kmeans=KMeans(n_clusters,random_state=random_state)
	gene_pred=kmeans.fit_predict(gene_exp_tmp)
	print("========== KMeans clustering results ==========")
	print("The final number of gene clusters is "+str(len(pd.Series(gene_pred).value_counts())))
	gene_adata.obs[cluster_key]=gene_pred.copy()
	gene_adata.obs[cluster_key]=gene_adata.obs[cluster_key].astype("category")
	return gene_adata


# gene - PCA/truncated SVD -> leiden clustering
# Brain Visium samples
def gene_PCA_clustering(gene_adata, gene_npcs, resolution, n_neighbors, cluster_key, nodes_num, random_state=0, approx_pca=True):
	# use truncated SVD
	if approx_pca is True:
		svd=TruncatedSVD(n_components=gene_npcs, n_iter=5, random_state=random_state) # n_iter default value is 5
		test_gene_pcs=svd.fit_transform(gene_adata.X)
	elif approx_pca is False:
		pca=PCA(n_components=gene_npcs, random_state=random_state)
		test_gene_pcs=pca.fit_transform(gene_adata.X)
	# leiden clustering
	gene_pred=leiden_clustering(features_matrix=test_gene_pcs, resolution=resolution, n_neighbors=n_neighbors, random_state=random_state, leiden_key=cluster_key)
	print("The final number of gene clusters is "+str(len(pd.Series(gene_pred).value_counts())))
	gene_adata.obs[cluster_key]=gene_pred.copy()
	return gene_adata


def hipt_pcs_clustering(hipt_adata, hipt_npcs, resolution, n_neighbors, cluster_key, nodes_num, random_state=0, large_clusters=True):
	return _hipt_pcs_clustering_impl(hipt_adata=hipt_adata, hipt_npcs=hipt_npcs, resolution=resolution, n_neighbors=n_neighbors, cluster_key=cluster_key, nodes_num=nodes_num, random_state=random_state, large_clusters=large_clusters, method="leiden")


# I + G clustering functions (kmeans clustering and leiden clustering)
def hipt_pcs_clustering_upd(hipt_adata, hipt_npcs, resolution, n_neighbors, cluster_key, nodes_num, n_clusters, random_state=0, large_clusters=True):
	if cluster_key=="leiden_clusters":
		return _hipt_pcs_clustering_impl(hipt_adata=hipt_adata, hipt_npcs=hipt_npcs, resolution=resolution, n_neighbors=n_neighbors, cluster_key=cluster_key, nodes_num=nodes_num, random_state=random_state, large_clusters=large_clusters, method="leiden")
	elif cluster_key=="kmeans_clusters":
		return _hipt_pcs_clustering_impl(hipt_adata=hipt_adata, hipt_npcs=hipt_npcs, resolution=resolution, n_neighbors=n_neighbors, cluster_key=cluster_key, nodes_num=nodes_num, random_state=random_state, large_clusters=large_clusters, method="kmeans", n_clusters=n_clusters)
	else:
		raise ValueError(f"Unsupported cluster_key for hipt_pcs_clustering_upd: {cluster_key}")
	return hipt_adata


def nn_detect(input_adata_annotated, input_adata_not_annotated, features_set, target_tissue_regions, anchor_key, metric="euclidean", label_col="label", knn=5):
	if metric not in ["euclidean", "cosine"]:
		raise ValueError("metric must be either 'euclidean' or 'cosine'")
	target_tissue_regions=_as_list(target_tissue_regions)
	adata_qry=input_adata_annotated.copy()
	adata_ref=input_adata_not_annotated.copy()
	#Define ds1 and ds2
	#ds1 (query)
	sub1=adata_qry[:,adata_qry.var.index.isin(features_set)]
	ds1=sub1.X[sub1.obs[label_col].isin(target_tissue_regions),:] #target_tissue_regions: a list containing target tissue regions.
	#ds2 (reference)
	sub2=adata_ref[:,adata_ref.var.index.isin(features_set)]
	ds2=sub2.X
	#Tree construction
	dim=ds2.shape[1]
	num_elements=ds2.shape[0]
	if(metric=="euclidean"):
		tree=hnswlib.Index(space="l2",dim=dim)
	elif(metric=="cosine"):
		tree=hnswlib.Index(space="cosine",dim=dim)
	#tree.init_index(max_elements=num_elements,ef_construction=200,M=16) # original version
	tree.init_index(max_elements=num_elements,ef_construction=200,M=16,random_seed=100)
	tree.set_ef(50)
	tree.add_items(data=ds2,num_threads=1) # num_threads default is -1
	#Identify nearest neighbors
	ind, distances=tree.knn_query(data=ds1,k=knn,num_threads=1) # num_threads default is -1
	match={}
	anchors=set()
	for a, b in zip(range(ds1.shape[0]), ind):
		nn=[]
		for b_ind, b_i in enumerate(b):
			match[(a,b_i)]=distances[a,b_ind]
			nn.append(b_i)
		anchors=anchors|set(nn)
	anchors=list(anchors)
	adata_ref.obs[anchor_key]=0
	if len(anchors)>0:
		adata_ref.obs.loc[adata_ref.obs.index[anchors], anchor_key]=1
	adata_ref.obs[anchor_key]=adata_ref.obs[anchor_key].astype("category")
	return adata_ref


def nn_based_anchor_detection(ref_adata_sca, test_adata_sca, combined_genes, target_regions, nontgt_regions, target_node, nontgt_node, label_col="label", knn=5, metric="euclidean"):
	target_regions=_as_list(target_regions)
	nontgt_regions=_as_list(nontgt_regions)
	input_adata_annotated=ref_adata_sca[ref_adata_sca.obs[label_col].isin(target_regions+nontgt_regions)].copy()
	nn_start_time=time.time()
	if isinstance(knn, int):
		# target regions
		test_adata_sca=nn_detect(input_adata_annotated=input_adata_annotated,
								 input_adata_not_annotated=test_adata_sca,
								 features_set=combined_genes,
								 target_tissue_regions=target_regions,
								 anchor_key=target_node+"_anchors",
								 metric=metric,
								 label_col=label_col,
								 knn=knn)
		# nontgt regions
		test_adata_sca=nn_detect(input_adata_annotated=input_adata_annotated,
								 input_adata_not_annotated=test_adata_sca,
								 features_set=combined_genes,
								 target_tissue_regions=nontgt_regions,
								 anchor_key=nontgt_node+"_anchors",
								 metric=metric,
								 label_col=label_col,
								 knn=knn)
	elif isinstance(knn, list):
		if len(knn)!=2:
			raise ValueError("knn list must have length 2: [target_knn, nontgt_knn]")
		print(knn)
		# target regions
		print("Adjusted KNN for target regions: "+str(knn[0]))
		test_adata_sca=nn_detect(input_adata_annotated=input_adata_annotated,
								 input_adata_not_annotated=test_adata_sca,
								 features_set=combined_genes,
								 target_tissue_regions=target_regions,
								 anchor_key=target_node+"_anchors",
								 metric=metric,
								 label_col=label_col,
								 knn=knn[0])
		# nontgt regions
		print("Adjusted KNN for nontgt regions: "+str(knn[1]))
		test_adata_sca=nn_detect(input_adata_annotated=input_adata_annotated,
								 input_adata_not_annotated=test_adata_sca,
								 features_set=combined_genes,
								 target_tissue_regions=nontgt_regions,
								 anchor_key=nontgt_node+"_anchors",
								 metric=metric,
								 label_col=label_col,
								 knn=knn[1])
	nn_end_time=time.time()
	nn_run_time=nn_end_time-nn_start_time
	print("=================================== Anchor Detection ===================================")
	print(f"The running time of nn anchor detection: {nn_run_time:.4f} seconds")
	print("\n")
	return test_adata_sca


def filter_ref_spots(ref_adata_sca, gene_list, regions, perct_cutoff=0.75, label_col="label"):
	spots_union_set=[]
	if len(gene_list)>0:
		for g in gene_list:
			if g not in ref_adata_sca.var.index.tolist():
				continue
			exp_tmp=ref_adata_sca[ref_adata_sca.obs[label_col].isin(regions),ref_adata_sca.var.index==g].copy()
			if exp_tmp.shape[0]==0:
				continue
			exp_bd=np.quantile(exp_tmp.X, perct_cutoff)
			g_spots=exp_tmp[exp_tmp.X>=exp_bd].obs.index.tolist()
			spots_union_set=list(set(g_spots) | set(spots_union_set))
		keep_perct=len(spots_union_set)/max(1, exp_tmp.shape[0])
		print("By applying >="+str(perct_cutoff)+" filtering in ["+", ".join(regions)+"]: "+str(round(keep_perct,2))+" ref spots were kept")
	else:
		print("No genes are provided for filtering ref spots in ["+", ".join(regions)+"]")
		spots_union_set=ref_adata_sca[ref_adata_sca.obs[label_col].isin(regions)].obs.index.tolist()
	return spots_union_set


def nn_based_anchor_detection_filter(ref_adata_sca, test_adata_sca, target_genes, nontgt_genes, target_regions, nontgt_regions, target_node, nontgt_node, label_col="label", knn=5, metric="euclidean", ref_spots_filtering=False, perct_cutoff=0.75):
	nn_start_time=time.time()
	target_regions=_as_list(target_regions)
	nontgt_regions=_as_list(nontgt_regions)
	if ref_spots_filtering==True:
		#--------------------- 1. filter ref spots ---------------------
		ref_counts=ref_adata_sca.obs[label_col].value_counts()
		target_spots_num=ref_counts.reindex(target_regions, fill_value=0).sum()
		nontgt_spots_num=ref_counts.reindex(nontgt_regions, fill_value=0).sum()
		# not apply ref spots filtering for a binary side that only take less than 1/knn proportion of the opposite side
		#--------------- target regions ---------------
		if (nontgt_spots_num>0) and ((target_spots_num/nontgt_spots_num)>=(1/knn)):
			target_spots_union_set=filter_ref_spots(ref_adata_sca=ref_adata_sca, gene_list=target_genes, regions=target_regions, perct_cutoff=perct_cutoff, label_col=label_col)
		else:
			target_spots_union_set=ref_adata_sca[ref_adata_sca.obs[label_col].isin(target_regions)].obs.index.tolist()
		#--------------- nontgt regions ---------------
		if (target_spots_num>0) and ((nontgt_spots_num/target_spots_num)>=(1/knn)):
			nontgt_spots_union_set=filter_ref_spots(ref_adata_sca=ref_adata_sca, gene_list=nontgt_genes, regions=nontgt_regions, perct_cutoff=perct_cutoff, label_col=label_col)
		else:
			nontgt_spots_union_set=ref_adata_sca[ref_adata_sca.obs[label_col].isin(nontgt_regions)].obs.index.tolist()
		# filter ref spots based on identified spots
		input_adata_annotated=ref_adata_sca[ref_adata_sca.obs.index.isin(target_spots_union_set+nontgt_spots_union_set)].copy()
	else:
		input_adata_annotated=ref_adata_sca[ref_adata_sca.obs[label_col].isin(target_regions+nontgt_regions)].copy()
	#--------------------- 2. detect anchors ---------------------
	combined_genes=target_genes+nontgt_genes
	if isinstance(knn, int):
		# target regions
		test_adata_sca=nn_detect(input_adata_annotated=input_adata_annotated,
								 input_adata_not_annotated=test_adata_sca,
								 features_set=combined_genes,
								 target_tissue_regions=target_regions,
								 anchor_key=target_node+"_anchors",
								 metric=metric,
								 label_col=label_col,
								 knn=knn)
		# nontgt regions
		test_adata_sca=nn_detect(input_adata_annotated=input_adata_annotated,
								 input_adata_not_annotated=test_adata_sca,
								 features_set=combined_genes,
								 target_tissue_regions=nontgt_regions,
								 anchor_key=nontgt_node+"_anchors",
								 metric=metric,
								 label_col=label_col,
								 knn=knn)
	elif isinstance(knn, list):
		print(knn)
		# target regions
		print("Adjusted KNN for target regions: "+str(knn[0]))
		test_adata_sca=nn_detect(input_adata_annotated=input_adata_annotated,
								 input_adata_not_annotated=test_adata_sca,
								 features_set=combined_genes,
								 target_tissue_regions=target_regions,
								 anchor_key=target_node+"_anchors",
								 metric=metric,
								 label_col=label_col,
								 knn=knn[0])
		# nontgt regions
		print("Adjusted KNN for nontgt regions: "+str(knn[1]))
		test_adata_sca=nn_detect(input_adata_annotated=input_adata_annotated,
								 input_adata_not_annotated=test_adata_sca,
								 features_set=combined_genes,
								 target_tissue_regions=nontgt_regions,
								 anchor_key=nontgt_node+"_anchors",
								 metric=metric,
								 label_col=label_col,
								 knn=knn[1])
	nn_end_time=time.time()
	nn_run_time=nn_end_time-nn_start_time
	print("=================================== Anchor Detection ===================================")
	print(f"The running time of nn anchor detection: {nn_run_time:.4f} seconds")
	print("\n")
	return test_adata_sca


def gene_thres_region_sections(ref_adata_sca_dic, ref_sections, gene, region, regions_dic, perct_cutoff, label_col):
	sections_kept=[]
	for section in ref_sections:
		if region in regions_dic[section]:
			ref_adata_sca=ref_adata_sca_dic[section]
			if gene not in ref_adata_sca.var.index.tolist():
				continue
			section_exp=ref_adata_sca.X[ref_adata_sca.obs[label_col]==region, ref_adata_sca.var.index==gene]
			section_exp_perct=_safe_quantile(section_exp, perct_cutoff, default=0)
			if section_exp_perct>0:
				sections_kept.append(section)
	return sections_kept


def lower_tail_thres(ref_adata_sca_dic, merged_ref_adata_sca, target_regions, opposite_regions, gene_list, perct_cf_upper=0.85, perct_cf_lower=0.15, label_col="label", merged_key="batch"):
	ref_sections=list(ref_adata_sca_dic.keys())
	regions_dic={}
	for section in ref_sections:
		section_regions=ref_adata_sca_dic[section].obs[label_col].value_counts().index.tolist()
		section_regions=[i for i in section_regions if i not in ["nan", "unknown"]]
		regions_dic[section]=section_regions
	# determine the lower tail threshold of each gene
	hier_thres_dic={}
	for gene in gene_list:
		d_r_s={}
		regions_kept=[]
		exp_bound=[]
		# determine the exp threshold from the opposite regions (across regions and sections) | perct_cutoff = 0.85 (perct_cf_upper)
		opposite_used=True
		perct_cutoff=perct_cf_upper
		for region in opposite_regions:
			sections_kept=gene_thres_region_sections(ref_adata_sca_dic, ref_sections, gene, region, regions_dic, perct_cutoff, label_col)
			if len(sections_kept)>0:
				d_r_s[region]=sections_kept
				regions_kept.append(region)
		# determine the exp threshold from the target regions (across regions and sections) | perct_cutoff = 1 - 0.85 = 0.15 (perct_cf_lower)
		if len(regions_kept)==0:
			opposite_used=False
			perct_cutoff=perct_cf_lower
			print(gene+" threshold = 0 in the opposite regions | perct_cf_upper = "+str(perct_cf_upper)+" -> Need to increase perct_cf_upper!")
			for region in target_regions:
				sections_kept=gene_thres_region_sections(ref_adata_sca_dic, ref_sections, gene, region, regions_dic, perct_cutoff, label_col)
				if len(sections_kept)>0:
					d_r_s[region]=sections_kept
					regions_kept.append(region)
		if len(regions_kept)==0:
			print(gene+" threshold = 0 in the target regions | perct_cf_lower = "+str(perct_cf_lower)+" -> Need to increase perct_cf_lower!")
		elif len(regions_kept)>0:
			print(gene+" included tissue regions: ["+", ".join(regions_kept)+"]")
			for region in regions_kept:
				if gene not in merged_ref_adata_sca.var.index.tolist():
					continue
				exp=merged_ref_adata_sca.X[(merged_ref_adata_sca.obs[label_col]==region) & (merged_ref_adata_sca.obs[merged_key].isin(d_r_s[region])), merged_ref_adata_sca.var.index==gene]
				exp_perct=_safe_quantile(exp, perct_cutoff, default=0)
				exp_bound.append(exp_perct)
			if len(exp_bound)>0:
				exp_thres=np.mean(exp_bound)
				hier_thres_dic[gene]=exp_thres
		print() # add a new line for each gene
	assert len(hier_thres_dic)>0, "Need to increase either perct_cf_upper or perct_cf_lower"
	return hier_thres_dic


# testified
def gene_anchor_detection(qry_adata_sca, hier_thres_dic):
	anchor_columns=[]
	for gene, hier_thres in hier_thres_dic.items():
		anchor_key="anchor_"+gene
		qry_adata_sca.obs[anchor_key]=0
		if gene not in qry_adata_sca.var.index.tolist():
			anchor_columns.append(anchor_key)
			continue
		adata_sub=qry_adata_sca[:,qry_adata_sca.var.index==gene]
		anchor_indices=adata_sub[(adata_sub.X>hier_thres)].obs.index.tolist()
		if len(anchor_indices)>0:
			qry_adata_sca.obs.loc[anchor_indices, anchor_key]=1
		anchor_columns.append(anchor_key)
	return qry_adata_sca, anchor_columns


def _hier_anchor_detection_impl(qry_adata_sca, hier_key, anchor_columns, max_p, thres_q, anchor_thres_upper_adjust=False, anchor_thres_lower_adjust=False, auto_adjust=False):
	_validate_obs_columns(qry_adata_sca, anchor_columns, context="query data")
	summary_key=hier_key+"_anchors_sum"
	qry_adata_sca.obs[summary_key]=qry_adata_sca.obs[anchor_columns].sum(axis=1)
	print(qry_adata_sca.obs[summary_key].value_counts())
	final_key=hier_key+"_anchors"
	summary_max=qry_adata_sca.obs[summary_key].max()
	if summary_max==0:
		anchor_thres=1
	else:
		anchor_thres=int(np.min([np.round(summary_max*max_p), qry_adata_sca.obs[summary_key].quantile(thres_q)]))
	anchor_thres=np.max([1, anchor_thres])
	print(hier_key+" hierarchy initial anchor counts threshold: "+str(anchor_thres))
	actual_p=anchor_thres/max(1, summary_max)
	if auto_adjust is True:
		anchor_thres_upper_adjust=anchor_thres_upper_adjust or (actual_p<=max_p)
		anchor_thres_lower_adjust=anchor_thres_lower_adjust or (actual_p>max_p)
	if anchor_thres_lower_adjust is True and summary_max>0:
		if actual_p>max_p:
			thres_q_upd=0.9
			print("============================ actual max value ratio is larger than "+str(max_p)+" -> lower bound adjustment with updated thres_q = "+str(thres_q_upd)+" ============================")
			anchor_thres=int(np.min([np.round(summary_max*max_p), qry_adata_sca.obs[summary_key].quantile(thres_q_upd)]))
	if anchor_thres_upper_adjust is True and summary_max>0:
		if actual_p<=max_p and qry_adata_sca.obs[summary_key].quantile(0.5)>=anchor_thres:
			max_p_upd=0.75
			print("============================ 50th quantile value is larger than "+str(anchor_thres)+" -> upper bound adjustment with updated max_p = "+str(max_p_upd)+" ============================")
			anchor_thres=int(np.min([np.round(summary_max*max_p_upd), qry_adata_sca.obs[summary_key].quantile(thres_q)]))
	anchor_thres=np.max([1, anchor_thres])
	print(hier_key+" hierarchy final anchor counts threshold: "+str(anchor_thres))
	qry_adata_sca.obs[final_key]=0
	anchor_indices=qry_adata_sca[(qry_adata_sca.obs[summary_key]>=anchor_thres)].obs.index.tolist()
	if len(anchor_indices)>0:
		qry_adata_sca.obs.loc[anchor_indices, final_key]=1
	return qry_adata_sca, anchor_thres


# testified
def hier_anchor_detection(qry_adata_sca, hier_key, anchor_columns, max_p, thres_q, anchor_thres_upper_adjust=False, anchor_thres_lower_adjust=False):
	return _hier_anchor_detection_impl(qry_adata_sca=qry_adata_sca, hier_key=hier_key, anchor_columns=anchor_columns, max_p=max_p, thres_q=thres_q, anchor_thres_upper_adjust=anchor_thres_upper_adjust, anchor_thres_lower_adjust=anchor_thres_lower_adjust, auto_adjust=False)


# automatically determine whether anchor_thres_upper_adjust and anchor_thres_lower_adjust or not
def hier_anchor_detection_upd(qry_adata_sca, hier_key, anchor_columns, max_p, thres_q, anchor_thres_upper_adjust=False, anchor_thres_lower_adjust=False):
	return _hier_anchor_detection_impl(qry_adata_sca=qry_adata_sca, hier_key=hier_key, anchor_columns=anchor_columns, max_p=max_p, thres_q=thres_q, anchor_thres_upper_adjust=anchor_thres_upper_adjust, anchor_thres_lower_adjust=anchor_thres_lower_adjust, auto_adjust=True)



# Range-based anchor detection
def range_based_anchor_detection(ref_adata_sca_dic, merged_ref_adata_sca, test_adata_sca, target_node, nontgt_node, target_regions, nontgt_regions, target_genes, nontgt_genes, perct_cf_upper, perct_cf_lower, max_p, thres_q, anchor_thres_upper_adjust, anchor_thres_lower_adjust, label_col, merged_key):
	# determine the gene exp threshold of two hierarchies
	target_hier_thres_dic=lower_tail_thres(ref_adata_sca_dic=ref_adata_sca_dic, 
									   	   merged_ref_adata_sca=merged_ref_adata_sca, 
									   	   target_regions=target_regions, 
									   	   opposite_regions=nontgt_regions,
									   	   gene_list=target_genes,
									   	   perct_cf_upper=perct_cf_upper,
									   	   perct_cf_lower=perct_cf_lower,
									   	   label_col=label_col,
									   	   merged_key=merged_key)
	print("=========== ["+", ".join(target_regions)+"] gene exp thres ===========")
	print(target_hier_thres_dic)
	nontgt_hier_thres_dic=lower_tail_thres(ref_adata_sca_dic=ref_adata_sca_dic, 
										   merged_ref_adata_sca=merged_ref_adata_sca, 
										   target_regions=nontgt_regions, 
										   opposite_regions=target_regions,
										   gene_list=nontgt_genes,
										   perct_cf_upper=perct_cf_upper,
										   perct_cf_lower=perct_cf_lower,
										   label_col=label_col,
										   merged_key=merged_key)
	print("=========== ["+", ".join(nontgt_regions)+"] gene exp thres ===========")
	print(nontgt_hier_thres_dic)
	# anchor detection
	test_adata_sca, target_anchor_columns=gene_anchor_detection(qry_adata_sca=test_adata_sca, hier_thres_dic=target_hier_thres_dic)
	#test_adata_sca=hier_anchor_detection(qry_adata_sca=test_adata_sca, hier_key=target_node, anchor_columns=target_anchor_columns, counts_ratio=counts_ratio)
	test_adata_sca, target_anchor_thres=hier_anchor_detection(qry_adata_sca=test_adata_sca, hier_key=target_node, anchor_columns=target_anchor_columns, max_p=max_p, thres_q=thres_q, anchor_thres_upper_adjust=anchor_thres_upper_adjust, anchor_thres_lower_adjust=anchor_thres_lower_adjust)
	test_adata_sca, nontgt_anchor_columns=gene_anchor_detection(qry_adata_sca=test_adata_sca, hier_thres_dic=nontgt_hier_thres_dic)
	#test_adata_sca=hier_anchor_detection(qry_adata_sca=test_adata_sca, hier_key=nontgt_node, anchor_columns=nontgt_anchor_columns, counts_ratio=counts_ratio)
	test_adata_sca, nontgt_anchor_thres=hier_anchor_detection(qry_adata_sca=test_adata_sca, hier_key=nontgt_node, anchor_columns=nontgt_anchor_columns, max_p=max_p, thres_q=thres_q, anchor_thres_upper_adjust=anchor_thres_upper_adjust, anchor_thres_lower_adjust=anchor_thres_lower_adjust)
	return test_adata_sca, target_anchor_thres, nontgt_anchor_thres


# Range-based anchor detection upd
# allow different anchor thresholding: max_p_list, thres_q_list
def range_based_anchor_detection_upd(ref_adata_sca_dic, merged_ref_adata_sca, test_adata_sca, target_node, nontgt_node, target_regions, nontgt_regions, target_genes, nontgt_genes, perct_cf_upper, perct_cf_lower, max_p_list, thres_q_list, anchor_thres_upper_adjust, anchor_thres_lower_adjust, label_col, merged_key):
	# determine the gene exp threshold of two hierarchies
	target_hier_thres_dic=lower_tail_thres(ref_adata_sca_dic=ref_adata_sca_dic, 
									   	   merged_ref_adata_sca=merged_ref_adata_sca, 
									   	   target_regions=target_regions, 
									   	   opposite_regions=nontgt_regions,
									   	   gene_list=target_genes,
									   	   perct_cf_upper=perct_cf_upper,
									   	   perct_cf_lower=perct_cf_lower,
									   	   label_col=label_col,
									   	   merged_key=merged_key)
	print("=========== ["+", ".join(target_regions)+"] gene exp thres ===========")
	print(target_hier_thres_dic)
	nontgt_hier_thres_dic=lower_tail_thres(ref_adata_sca_dic=ref_adata_sca_dic, 
										   merged_ref_adata_sca=merged_ref_adata_sca, 
										   target_regions=nontgt_regions, 
										   opposite_regions=target_regions,
										   gene_list=nontgt_genes,
										   perct_cf_upper=perct_cf_upper,
										   perct_cf_lower=perct_cf_lower,
										   label_col=label_col,
										   merged_key=merged_key)
	print("=========== ["+", ".join(nontgt_regions)+"] gene exp thres ===========")
	print(nontgt_hier_thres_dic)
	# anchor detection
	test_adata_sca, target_anchor_columns=gene_anchor_detection(qry_adata_sca=test_adata_sca, hier_thres_dic=target_hier_thres_dic)
	#test_adata_sca=hier_anchor_detection(qry_adata_sca=test_adata_sca, hier_key=target_node, anchor_columns=target_anchor_columns, counts_ratio=counts_ratio)
	test_adata_sca, target_anchor_thres=hier_anchor_detection(qry_adata_sca=test_adata_sca, hier_key=target_node, anchor_columns=target_anchor_columns, max_p=max_p_list[0], thres_q=thres_q_list[0], anchor_thres_upper_adjust=anchor_thres_upper_adjust, anchor_thres_lower_adjust=anchor_thres_lower_adjust)
	test_adata_sca, nontgt_anchor_columns=gene_anchor_detection(qry_adata_sca=test_adata_sca, hier_thres_dic=nontgt_hier_thres_dic)
	#test_adata_sca=hier_anchor_detection(qry_adata_sca=test_adata_sca, hier_key=nontgt_node, anchor_columns=nontgt_anchor_columns, counts_ratio=counts_ratio)
	test_adata_sca, nontgt_anchor_thres=hier_anchor_detection(qry_adata_sca=test_adata_sca, hier_key=nontgt_node, anchor_columns=nontgt_anchor_columns, max_p=max_p_list[1], thres_q=thres_q_list[1], anchor_thres_upper_adjust=anchor_thres_upper_adjust, anchor_thres_lower_adjust=anchor_thres_lower_adjust)
	return test_adata_sca, target_anchor_thres, nontgt_anchor_thres


# Range-based anchor detection most upd version (enabled automatically detection of upper and lower anchor thres adjustments)
def range_based_anchor_detection_upd_v2(ref_adata_sca_dic, merged_ref_adata_sca, test_adata_sca, target_node, nontgt_node, target_regions, nontgt_regions, target_genes, nontgt_genes, perct_cf_upper, perct_cf_lower, max_p_list, thres_q_list, anchor_thres_upper_adjust, anchor_thres_lower_adjust, label_col, merged_key):
	# determine the gene exp threshold of two hierarchies
	target_hier_thres_dic=lower_tail_thres(ref_adata_sca_dic=ref_adata_sca_dic, 
									   	   merged_ref_adata_sca=merged_ref_adata_sca, 
									   	   target_regions=target_regions, 
									   	   opposite_regions=nontgt_regions,
									   	   gene_list=target_genes,
									   	   perct_cf_upper=perct_cf_upper,
									   	   perct_cf_lower=perct_cf_lower,
									   	   label_col=label_col,
									   	   merged_key=merged_key)
	print("=========== ["+", ".join(target_regions)+"] gene exp thres ===========")
	print(target_hier_thres_dic)
	nontgt_hier_thres_dic=lower_tail_thres(ref_adata_sca_dic=ref_adata_sca_dic, 
										   merged_ref_adata_sca=merged_ref_adata_sca, 
										   target_regions=nontgt_regions, 
										   opposite_regions=target_regions,
										   gene_list=nontgt_genes,
										   perct_cf_upper=perct_cf_upper,
										   perct_cf_lower=perct_cf_lower,
										   label_col=label_col,
										   merged_key=merged_key)
	print("=========== ["+", ".join(nontgt_regions)+"] gene exp thres ===========")
	print(nontgt_hier_thres_dic)
	# anchor detection
	test_adata_sca, target_anchor_columns=gene_anchor_detection(qry_adata_sca=test_adata_sca, hier_thres_dic=target_hier_thres_dic)
	#test_adata_sca=hier_anchor_detection(qry_adata_sca=test_adata_sca, hier_key=target_node, anchor_columns=target_anchor_columns, counts_ratio=counts_ratio)
	test_adata_sca, target_anchor_thres=hier_anchor_detection_upd(qry_adata_sca=test_adata_sca, hier_key=target_node, anchor_columns=target_anchor_columns, max_p=max_p_list[0], thres_q=thres_q_list[0], anchor_thres_upper_adjust=anchor_thres_upper_adjust, anchor_thres_lower_adjust=anchor_thres_lower_adjust)
	test_adata_sca, nontgt_anchor_columns=gene_anchor_detection(qry_adata_sca=test_adata_sca, hier_thres_dic=nontgt_hier_thres_dic)
	#test_adata_sca=hier_anchor_detection(qry_adata_sca=test_adata_sca, hier_key=nontgt_node, anchor_columns=nontgt_anchor_columns, counts_ratio=counts_ratio)
	test_adata_sca, nontgt_anchor_thres=hier_anchor_detection_upd(qry_adata_sca=test_adata_sca, hier_key=nontgt_node, anchor_columns=nontgt_anchor_columns, max_p=max_p_list[1], thres_q=thres_q_list[1], anchor_thres_upper_adjust=anchor_thres_upper_adjust, anchor_thres_lower_adjust=anchor_thres_lower_adjust)
	return test_adata_sca, target_anchor_thres, nontgt_anchor_thres


def _build_anchor_cross_table(adata, hier_index, hier_anchor_key, cluster_key, weight_mode="plain", anchor_weight_g=1, anchor_weight_p=1):
	_validate_obs_columns(adata, [cluster_key] + hier_anchor_key, context="input data")
	clusters=adata.obs[cluster_key].value_counts().index.tolist()
	cluster_names=["cluster_"+str(i) for i in clusters]
	cross_table=pd.DataFrame(np.zeros((len(hier_index),len(clusters))),index=hier_index,columns=cluster_names)
	for i in clusters:
		cluster_sub=adata[adata.obs[cluster_key]==i]
		cluster_size=max(1, cluster_sub.shape[0])
		for j in range(len(hier_index)):
			anchor_series=cluster_sub.obs[hier_anchor_key[j]]
			if weight_mode=="plain":
				anchor_spots=anchor_series.value_counts()
				if 1 in anchor_spots.index.tolist():
					cross_table.loc[hier_index[j],"cluster_"+str(i)]=anchor_spots.loc[1]/cluster_size
			elif weight_mode=="weighted":
				cross_table.loc[hier_index[j],"cluster_"+str(i)]=anchor_series.sum()/cluster_size
			elif weight_mode=="weighted_multi":
				cross_table.loc[hier_index[j],"cluster_"+str(i)]=anchor_series.sum()/(cluster_size*(anchor_weight_g+anchor_weight_p))
			else:
				raise ValueError(f"Unsupported weight_mode: {weight_mode}")
	return cross_table*100


def _assign_from_cross_table(adata, cross_table, hier_index, cluster_key, infer_key, drop_thres, nan_thres, prop_diff_cutoff=1, allow_novel_clusters=False, cluster_spots_prop=None, num_nbs=15, x_col="x", y_col="y"):
	cluster_spots=adata.obs[cluster_key].value_counts()
	keep_clusters=cluster_spots[cluster_spots>=drop_thres].index.tolist()
	drop_clusters=[i for i in cluster_spots.index.tolist() if i not in keep_clusters]
	if len(drop_clusters)>0:
		print("The dropped clusters: "+", ".join(map(str, drop_clusters)))
	adata=adata[adata.obs[cluster_key].isin(keep_clusters)].copy()
	cross_table=cross_table.loc[:, [c for c in cross_table.columns if int(str(c).replace("cluster_", "")) in keep_clusters]]
	max_col=cross_table.idxmax()
	same_prop_clusters=cross_table.columns[cross_table.iloc[0]==cross_table.iloc[1]].tolist() if cross_table.shape[0] >= 2 else []
	if len(same_prop_clusters)>0:
		print("The clusters having the same proportions across two hierarchy anchors: "+", ".join(same_prop_clusters))
	prop_diff=abs(cross_table.iloc[0,:]-cross_table.iloc[1,:]).sort_values() if cross_table.shape[0] >= 2 else pd.Series(dtype=float)
	print("=========== The absolute proportion difference across two hierarchies ===========")
	print(prop_diff)
	similar_clusters=prop_diff[prop_diff<=prop_diff_cutoff].index.tolist() if len(prop_diff)>0 else []
	cross_table_upd=cross_table
	max_col_upd=max_col
	if len(similar_clusters)>0:
		similar_clusters=_safe_cluster_ids_from_index(similar_clusters)
		print("The clusters having the similar proportions across two hierarchy anchors: "+", ".join(map(str,similar_clusters)))
		cross_table_upd=cross_table.copy()
		for i in range(len(hier_index)):
			hier_sub=adata[adata.obs[hier_anchor_key[i]]==1].copy()
			print("------------------- "+hier_index[i]+" -------------------")
			print("The number of detected anchors: "+str(hier_sub.shape[0]))
			hier_cluster_prop=hier_sub.obs[cluster_key].value_counts(normalize=True)
			hier_cluster_spots=hier_sub.obs[cluster_key].value_counts()
			hier_cluster_summary=pd.DataFrame({"spots_num": pd.Series(hier_cluster_spots), "percentage": pd.Series(hier_cluster_prop)})
			print("The anchors distribution across clusters")
			print(hier_cluster_summary)
			for j in similar_clusters:
				if j in hier_cluster_prop.index.tolist():
					cross_table_upd.loc[hier_index[i], "cluster_"+str(j)]=hier_cluster_prop[j]*cross_table.loc[hier_index[i], "cluster_"+str(j)]
		print("========== Updated Cross Table of Anchors (after assigning the weights of anchors that fall in each cluster) ==========")
		print(cross_table_upd)
		max_col_upd=cross_table_upd.idxmax()
		same_prop_clusters_upd=cross_table_upd.columns[cross_table_upd.iloc[0]==cross_table_upd.iloc[1]].tolist() if cross_table_upd.shape[0] >= 2 else []
		if len(same_prop_clusters_upd)>0:
			print("\n")
			print("After adjusted by weights, the clusters having the same proportions of two hierarchy anchors: "+", ".join(same_prop_clusters_upd))
		max_col_summary=pd.DataFrame({'original': pd.Series(max_col), 'adjusted_by_weights': pd.Series(max_col_upd)})
		print("=========== Clusters Label Assignment Difference (w/wth weights adjustment) ===========")
		print(max_col_summary)
		diff_clusters=max_col_summary.index[max_col_summary.iloc[:,0]!=max_col_summary.iloc[:,1]].tolist()
		if len(diff_clusters)>0:
			print("After adjusted by weights, the clusters that have different label assignments: "+", ".join(diff_clusters))
		else:
			print("The label assignments of clusters keep the same after adjusted by weights")
	adata.obs[infer_key]="novel_cluster"
	adata.obs[infer_key]=adata.obs[infer_key].astype(str)
	novel_clusters=[]
	for j in range(len(hier_index)):
		active_max_col=max_col_upd if len(similar_clusters)>0 else max_col
		cluster_index=active_max_col.index[active_max_col==hier_index[j]].tolist()
		cluster_prop=cross_table.loc[hier_index[j],cluster_index] if len(cluster_index)>0 else pd.Series(dtype=float)
		cluster_index=cluster_prop[cluster_prop>nan_thres].index.tolist() if len(cluster_prop)>0 else []
		novel_index=cluster_prop[cluster_prop<=nan_thres].index.tolist() if len(cluster_prop)>0 else []
		novel_clusters=novel_clusters+[int(str(k).replace("cluster_", "")) for k in novel_index]
		match_clusters=[int(str(k).replace("cluster_", "")) for k in cluster_index]
		if len(match_clusters)>0:
			adata.obs.loc[adata.obs[cluster_key].isin(match_clusters),infer_key]=hier_index[j]
	if len(novel_clusters)>0:
		adata.obs.loc[adata.obs[cluster_key].isin(novel_clusters),infer_key]="novel_cluster"
		print("novel clusters: "+", ".join(map(str, novel_clusters)))
		print("========== Before Reassigning Novel Clusters ==========")
		print(adata.obs[infer_key].value_counts())
		if allow_novel_clusters is True and cluster_spots_prop is not None:
			novel_clusters_prop=cluster_spots_prop.reindex(novel_clusters).fillna(0)
			novel_clusters=novel_clusters_prop[novel_clusters_prop>0.1].index.tolist()
			print("Updated novel clusters: "+",".join(map(str, novel_clusters)))
		for novel_i in novel_clusters:
			adata=reassign_novel_cluster(input_adata=adata, novel_i=novel_i, cluster_key=cluster_key, infer_key=infer_key, x_col=x_col, y_col=y_col, num_nbs=num_nbs)
	adata.obs[infer_key]=adata.obs[infer_key].astype("category")
	print("========== Inferred Labels ==========")
	print(adata.obs[infer_key].value_counts())
	return adata, cross_table, cross_table_upd


# most updated one: fix one bug
def hier_assign_labels(input_adata, hier_index=["a","b"], hier_anchor_key=["anchor_a","anchor_b"], infer_key="infer_label", cluster_key="louvain_clusters", drop_thres=100, nan_thres=5):
	adata=input_adata.copy()
	cross_table=_build_anchor_cross_table(adata=adata, hier_index=hier_index, hier_anchor_key=hier_anchor_key, cluster_key=cluster_key, weight_mode="plain")
	print("========== Cross Table of Anchors ==========")
	print(cross_table.round(2))
	adata.obs[infer_key]="nan"
	adata.obs[infer_key]=adata.obs[infer_key].astype(str)
	cluster_spots=adata.obs[cluster_key].value_counts()
	nan_clusters=cluster_spots[cluster_spots<drop_thres].index.tolist()
	drop_cluster_index=["cluster_"+str(i) for i in nan_clusters]
	cross_table_upd=cross_table.drop(columns=drop_cluster_index, errors="ignore")
	max_col=cross_table_upd.idxmax() if cross_table_upd.shape[1]>0 else pd.Series(dtype=object)
	for j in range(len(hier_index)):
		cluster_index=max_col.index[max_col==hier_index[j]].tolist() if len(max_col)>0 else []
		cluster_prop=cross_table_upd.loc[hier_index[j],cluster_index] if len(cluster_index)>0 else pd.Series(dtype=float)
		cluster_index=cluster_prop[cluster_prop>nan_thres].index.tolist() if len(cluster_prop)>0 else []
		match_clusters=[int(str(k).replace("cluster_", "")) for k in cluster_index]
		if len(match_clusters)>0:
			adata.obs.loc[adata.obs[cluster_key].isin(match_clusters),infer_key]=hier_index[j]
	adata.obs[infer_key]=adata.obs[infer_key].astype("category")
	print("========== Inferred Labels ==========")
	print(adata.obs[infer_key].value_counts())
	return adata, cross_table_upd



# designed for multi-modality anchor detection case (G + P - anchor detection)
# untestified 
def hier_assign_labels_weights_multi(input_adata, hier_index=["a","b"], hier_anchor_key=["a_anchors","b_anchors"], infer_key="a_vs_b", cluster_key="leiden_clusters",x_col="x", y_col="y", allow_novel_clusters=False, anchor_weight_g=1, anchor_weight_p=1, drop_thres=100, nan_thres=5, prop_diff_cutoff=1, num_nbs=25):
	adata=input_adata.copy()
	cluster_spots_prop=adata.obs[cluster_key].value_counts(normalize=True)
	print(cluster_spots_prop)
	cross_table=_build_anchor_cross_table(adata=adata, hier_index=hier_index, hier_anchor_key=hier_anchor_key, cluster_key=cluster_key, weight_mode="weighted_multi", anchor_weight_g=anchor_weight_g, anchor_weight_p=anchor_weight_p)
	print("========== Cross Table of Anchors ==========")
	print(cross_table)
	adata, _, _= _assign_from_cross_table(adata=adata, cross_table=cross_table, hier_index=hier_index, cluster_key=cluster_key, infer_key=infer_key, drop_thres=drop_thres, nan_thres=nan_thres, prop_diff_cutoff=prop_diff_cutoff, allow_novel_clusters=allow_novel_clusters, cluster_spots_prop=cluster_spots_prop, num_nbs=num_nbs, x_col=x_col, y_col=y_col)
	return adata


# testified
# most updated version: address novel cluster assignment issues (12/25/2024)
def hier_assign_labels_weights(input_adata, hier_index=["a","b"], hier_anchor_key=["a_anchors","b_anchors"], infer_key="a_vs_b", cluster_key="leiden_clusters", x_col="x", y_col="y", allow_novel_clusters=False, drop_thres=100, nan_thres=5, prop_diff_cutoff=1, num_nbs=25):
	adata=input_adata.copy()
	cluster_spots_prop=adata.obs[cluster_key].value_counts(normalize=True)
	print(cluster_spots_prop)
	cross_table=_build_anchor_cross_table(adata=adata, hier_index=hier_index, hier_anchor_key=hier_anchor_key, cluster_key=cluster_key, weight_mode="weighted")
	print("========== Cross Table of Anchors ==========")
	print(cross_table)
	adata, _, _ = _assign_from_cross_table(adata=adata, cross_table=cross_table, hier_index=hier_index, cluster_key=cluster_key, infer_key=infer_key, drop_thres=drop_thres, nan_thres=nan_thres, prop_diff_cutoff=prop_diff_cutoff, allow_novel_clusters=allow_novel_clusters, cluster_spots_prop=cluster_spots_prop, num_nbs=num_nbs, x_col=x_col, y_col=y_col)
	return adata


# testified
# novel cluster assignment
def reassign_novel_cluster(input_adata, novel_i, cluster_key, infer_key, x_col, y_col, num_nbs=15):
	#----------------------- 1. calculate the pairwise distances between each novel cluster and its neighboring clusters -----------------------
	# novel cluster
	novel_df=input_adata[input_adata.obs[cluster_key]==novel_i].obs.copy()
	novel_pred=pd.DataFrame({"pred": novel_df[infer_key]}, index=novel_df.index.tolist())
	novel_coords=novel_df[[x_col,y_col]]
	# other clusters
	other_df=input_adata[input_adata.obs[infer_key]!="novel_cluster"].obs.copy()
	other_pred=pd.DataFrame({"pred": other_df[infer_key]}, index=other_df.index.tolist())
	other_coords=other_df[[x_col,y_col]] 
	if other_df.shape[0]==0 or novel_df.shape[0]==0:
		print("---------------- novel cluster "+str(novel_i)+" ----------------")
		print("No valid reassignment neighborhood found; keeping existing label")
		return input_adata
	# calculate distances
	dists=cdist(np.array(novel_coords), np.array(other_coords), metric="euclidean")
	radius_perct=min(1.0, max(1, num_nbs)/max(1, other_df.shape[0]))
	radius=np.quantile(dists.flatten(),radius_perct)
	dists=pd.DataFrame(dists, index=novel_df.index.tolist(), columns=other_df.index.tolist())
	#----------------------- 2. reassign novel cluster identities -----------------------
	# the identity was reassigned to the neighboring cluster which takes the largest proportion in the neighborhood of each novel cluster
	novel_nbs_pred=[]
	for i in range(dists.shape[0]):
		index=dists.index.tolist()[i]
		dists_tmp=dists.loc[index, :].sort_values()
		nbs=dists_tmp[dists_tmp<radius]
		#nbs=dists_tmp[0:num_nbs]
		if len(nbs)>0:
			nbs_pred=other_pred.loc[nbs.index, "pred"]
			novel_nbs_pred=novel_nbs_pred+nbs_pred.tolist()
	if len(novel_nbs_pred)==0:
		print("---------------- novel cluster "+str(novel_i)+" ----------------")
		print("No neighbors fell within the reassignment radius; keeping novel_cluster")
		return input_adata
	refined_novel_pred=pd.Series(novel_nbs_pred).value_counts().idxmax()
	print("---------------- novel cluster "+str(novel_i)+" ----------------")
	print(pd.Series(novel_nbs_pred).value_counts())
	print("Based on neighborhood composition, novel cluster "+str(novel_i)+" reassignment: "+refined_novel_pred)
	# refine the novel cluster identities
	input_adata.obs.loc[novel_df.index.tolist(),infer_key]=refined_novel_pred
	return input_adata


def _collect_edge_features_from_dicts(dict_items, target_node, nontgt_node, sections_used=None):
	target_features=[]
	nontgt_features=[]
	target_key=target_node+"_vs_"+nontgt_node
	nontgt_key=nontgt_node+"_vs_"+target_node
	for section_name, feature_dic in dict_items:
		if (sections_used is not None) and (section_name not in sections_used):
			continue
		if target_key in feature_dic:
			target_features.extend(_as_list(feature_dic[target_key]))
		if nontgt_key in feature_dic:
			nontgt_features.extend(_as_list(feature_dic[nontgt_key]))
	return _unique_preserve_order(target_features), _unique_preserve_order(nontgt_features)


def _build_combined_matrix(adata_parts, feature_lists):
	mats=[]
	for adata, feats in zip(adata_parts, feature_lists):
		if (adata is None) or (feats is None) or (len(feats)==0):
			continue
		feats=[f for f in feats if f in adata.var.index.tolist()]
		if len(feats)==0:
			continue
		mats.append(adata[:,adata.var.index.isin(feats)].X)
	if len(mats)==0:
		raise ValueError("No usable modality features found for combined clustering")
	if len(mats)==1:
		return mats[0]
	return np.hstack(mats)


def _set_output_labels(state, infer_key, labels):
	for key in ["test_gene_sca", "test_hipt", "test_gene", "test_protein_sca", "test_protein"]:
		if key in state and state[key] is not None:
			state[key].obs[infer_key]=labels.copy()
			state[key].obs[infer_key]=state[key].obs[infer_key].astype("category")


def _cluster_state(state, config, target_node, nontgt_node, target_genes, nontgt_genes, target_image=None, nontgt_image=None, target_protein=None, nontgt_protein=None):
	cluster_key=state.get("cluster_key", "leiden_clusters")
	cluster_backend=config.get("cluster_backend", "leiden")
	cluster_source=config.get("cluster_source", "gene")
	cluster_outputs=config.get("cluster_outputs", ["test_gene_sca"])
	hipt_npcs=config.get("hipt_npcs", 10)
	gene_npcs=config.get("gene_npcs", 30)
	resolution=config.get("resolution", 0.2)
	n_neighbors=config.get("n_neighbors", 10)
	n_clusters=config.get("n_clusters", 5)
	random_state=config.get("random_state", 0)
	boundary_mode=config.get("boundary_mode", "none")
	large_clusters=config.get("large_clusters", True)
	if cluster_source=="gene":
		if config.get("feature_transform", "raw")=="pca":
			state["test_gene"]=gene_PCA_clustering(gene_adata=state["test_gene"], gene_npcs=gene_npcs, resolution=resolution, n_neighbors=n_neighbors, cluster_key=cluster_key, nodes_num=len(state["node_dic"][state["root_node"]]), random_state=random_state, approx_pca=True)
		else:
			if cluster_backend=="kmeans":
				state["test_gene"]=gene_clustering_kmeans(gene_adata=state["test_gene"], gene_features=target_genes+nontgt_genes, n_clusters=n_clusters, cluster_key=cluster_key, random_state=random_state)
			else:
				state["test_gene"]=gene_clustering(gene_adata=state["test_gene"], gene_features=target_genes+nontgt_genes, resolution=resolution, n_neighbors=n_neighbors, cluster_key=cluster_key, nodes_num=len(state["node_dic"][state["root_node"]]), random_state=random_state, large_clusters=large_clusters)
		labels=state["test_gene"].obs[cluster_key].tolist()
	elif cluster_source=="hipt":
		if cluster_backend=="kmeans":
			state["test_hipt"]=hipt_pcs_clustering_upd(hipt_adata=state["test_hipt"], hipt_npcs=hipt_npcs, resolution=resolution, n_neighbors=n_neighbors, cluster_key=cluster_key, nodes_num=len(state["node_dic"][state["root_node"]]), n_clusters=n_clusters, random_state=random_state, large_clusters=large_clusters)
		else:
			state["test_hipt"]=hipt_pcs_clustering(hipt_adata=state["test_hipt"], hipt_npcs=hipt_npcs, resolution=resolution, n_neighbors=n_neighbors, cluster_key=cluster_key, nodes_num=len(state["node_dic"][state["root_node"]]), random_state=random_state, large_clusters=large_clusters)
		bd_cluster=None
		if boundary_mode in ["identify", "reassign", "auto"]:
			bd_cluster=identify_bd_cluster_upd(hipt_adata=state["test_hipt"], cluster_key=cluster_key)
			if boundary_mode=="reassign":
				state["test_hipt"], refined_bd_key, final_cluster_key=reassign_bd_cluster(hipt_adata=state["test_hipt"], bd_cluster=bd_cluster, cluster_key=cluster_key, x_col=config.get("x_col", "x"), y_col=config.get("y_col", "y"), bd_num_nbs_1=config.get("bd_num_nbs_1", 25), bd_num_nbs_2=config.get("bd_num_nbs_2", 15))
				labels=state["test_hipt"].obs[final_cluster_key].tolist()
				cluster_key=final_cluster_key
			else:
				labels=state["test_hipt"].obs[cluster_key].tolist()
		else:
			labels=state["test_hipt"].obs[cluster_key].tolist()
	elif cluster_source=="combined":
		cluster_features=_build_combined_matrix(
			adata_parts=[state.get("test_gene"), state.get("test_hipt"), state.get("test_protein")],
			feature_lists=[target_genes+nontgt_genes, _as_list(target_image)+_as_list(nontgt_image), _as_list(target_protein)+_as_list(nontgt_protein)],
		)
		if cluster_backend=="kmeans":
			labels=kmeans_clustering(features_matrix=cluster_features, n_clusters=n_clusters, random_state=random_state, kmeans_key=cluster_key)
		else:
			labels=leiden_clustering(features_matrix=cluster_features, resolution=resolution, n_neighbors=n_neighbors, random_state=random_state, leiden_key=cluster_key)
	else:
		raise ValueError(f"Unsupported cluster_source: {cluster_source}")
	for key in cluster_outputs:
		if key=="test_gene":
			state["test_gene"].obs[cluster_key]=labels.copy()
			state["test_gene"].obs[cluster_key]=state["test_gene"].obs[cluster_key].astype("category")
		elif key=="test_gene_sca":
			state["test_gene_sca"].obs[cluster_key]=labels.copy()
			state["test_gene_sca"].obs[cluster_key]=state["test_gene_sca"].obs[cluster_key].astype("category")
		elif key=="test_hipt":
			state["test_hipt"].obs[cluster_key]=labels.copy()
			state["test_hipt"].obs[cluster_key]=state["test_hipt"].obs[cluster_key].astype("category")
		elif key=="test_protein":
			state["test_protein"].obs[cluster_key]=labels.copy()
			state["test_protein"].obs[cluster_key]=state["test_protein"].obs[cluster_key].astype("category")
		elif key=="test_protein_sca":
			state["test_protein_sca"].obs[cluster_key]=labels.copy()
			state["test_protein_sca"].obs[cluster_key]=state["test_protein_sca"].obs[cluster_key].astype("category")
	print("=================== Clustering results ===================")
	if "test_gene_sca" in state and state["test_gene_sca"] is not None:
		print(state["test_gene_sca"].obs[cluster_key].value_counts())
	return state, cluster_key


def _detect_anchors_state(state, config, target_node, nontgt_node, target_regions, nontgt_regions, target_genes, nontgt_genes, target_protein=None, nontgt_protein=None):
	anchor_mode=config.get("anchor_mode", "nn")
	label_col=config.get("label_col", "label")
	ref_spots_filtering=config.get("ref_spots_filtering", False)
	knn=config.get("knn", 5)
	metric=config.get("metric", "euclidean")
	if anchor_mode=="nn":
		if config.get("reference_mode", "single")=="multi":
			if config.get("modality_mode")=="gene_image_protein":
				raise NotImplementedError("multi-ref protein anchor detection is not enabled in the canonical driver")
			for ref_payload in state["ref_payloads"]:
				section_name=ref_payload["name"]
				if section_name not in state.get("sections_used", [section_name]):
					continue
				target_node_name=section_name+"_"+target_node
				nontgt_node_name=section_name+"_"+nontgt_node
				if ref_spots_filtering:
					state["test_gene_sca"]=nn_based_anchor_detection_filter(ref_adata_sca=ref_payload["gene"], test_adata_sca=state["test_gene_sca"], target_genes=target_genes, nontgt_genes=nontgt_genes, target_regions=target_regions, nontgt_regions=nontgt_regions, target_node=target_node_name, nontgt_node=nontgt_node_name, label_col=label_col, knn=knn, metric=metric, ref_spots_filtering=ref_spots_filtering, perct_cutoff=config.get("perct_cutoff", 0.75))
				else:
					state["test_gene_sca"]=nn_based_anchor_detection(ref_adata_sca=ref_payload["gene"], test_adata_sca=state["test_gene_sca"], combined_genes=target_genes+nontgt_genes, target_regions=target_regions, nontgt_regions=nontgt_regions, target_node=target_node_name, nontgt_node=nontgt_node_name, label_col=label_col, knn=knn, metric=metric)
			target_anchor_sum=np.zeros(state["test_gene_sca"].shape[0], dtype=int)
			nontgt_anchor_sum=np.zeros(state["test_gene_sca"].shape[0], dtype=int)
			for ref_payload in state["ref_payloads"]:
				section_name=ref_payload["name"]
				if section_name not in state.get("sections_used", [section_name]):
					continue
				target_anchor_sum=target_anchor_sum+state["test_gene_sca"].obs[section_name+"_"+target_node+"_anchors"].astype(int).to_numpy()
				nontgt_anchor_sum=nontgt_anchor_sum+state["test_gene_sca"].obs[section_name+"_"+nontgt_node+"_anchors"].astype(int).to_numpy()
			state["test_gene_sca"].obs[target_node+"_anchors_sum"]=target_anchor_sum
			state["test_gene_sca"].obs[nontgt_node+"_anchors_sum"]=nontgt_anchor_sum
			state["test_gene_sca"].obs[target_node+"_anchors"]=(state["test_gene_sca"].obs[target_node+"_anchors_sum"]>0).astype(int)
			state["test_gene_sca"].obs[nontgt_node+"_anchors"]=(state["test_gene_sca"].obs[nontgt_node+"_anchors_sum"]>0).astype(int)
		else:
			if config.get("modality_mode")=="gene_image_protein":
				state["test_gene_sca"]=nn_based_anchor_detection_filter(ref_adata_sca=state["ref_gene_sca"], test_adata_sca=state["test_gene_sca"], target_genes=target_genes, nontgt_genes=nontgt_genes, target_regions=target_regions, nontgt_regions=nontgt_regions, target_node=target_node, nontgt_node=nontgt_node, label_col=label_col, knn=knn, metric=metric, ref_spots_filtering=ref_spots_filtering, perct_cutoff=config.get("perct_cutoff", 0.75))
				state["test_protein_sca"]=nn_based_anchor_detection_filter(ref_adata_sca=state["ref_protein_sca"], test_adata_sca=state["test_protein_sca"], target_genes=target_protein, nontgt_genes=nontgt_protein, target_regions=target_regions, nontgt_regions=nontgt_regions, target_node=target_node+"_protein", nontgt_node=nontgt_node+"_protein", label_col=label_col, knn=knn, metric=metric, ref_spots_filtering=ref_spots_filtering, perct_cutoff=config.get("perct_cutoff", 0.75))
				state["test_gene_sca"].obs[target_node+"_gene_anchors"]=state["test_gene_sca"].obs[target_node+"_anchors"].copy()
				state["test_gene_sca"].obs[nontgt_node+"_gene_anchors"]=state["test_gene_sca"].obs[nontgt_node+"_anchors"].copy()
				state["test_protein_sca"].obs[target_node+"_protein_anchors"]=state["test_protein_sca"].obs[target_node+"_anchors"].copy()
				state["test_protein_sca"].obs[nontgt_node+"_protein_anchors"]=state["test_protein_sca"].obs[nontgt_node+"_anchors"].copy()
				state["test_gene_sca"].obs[target_node+"_anchors"]=state["test_gene_sca"].obs[target_node+"_gene_anchors"].astype(int)*config.get("anchor_weight_g", 1)+state["test_protein_sca"].obs[target_node+"_protein_anchors"].astype(int)*config.get("anchor_weight_p", 1)
				state["test_gene_sca"].obs[nontgt_node+"_anchors"]=state["test_gene_sca"].obs[nontgt_node+"_gene_anchors"].astype(int)*config.get("anchor_weight_g", 1)+state["test_protein_sca"].obs[nontgt_node+"_protein_anchors"].astype(int)*config.get("anchor_weight_p", 1)
			else:
				if ref_spots_filtering:
					state["test_gene_sca"]=nn_based_anchor_detection_filter(ref_adata_sca=state["ref_gene_sca"], test_adata_sca=state["test_gene_sca"], target_genes=target_genes, nontgt_genes=nontgt_genes, target_regions=target_regions, nontgt_regions=nontgt_regions, target_node=target_node, nontgt_node=nontgt_node, label_col=label_col, knn=knn, metric=metric, ref_spots_filtering=ref_spots_filtering, perct_cutoff=config.get("perct_cutoff", 0.75))
				else:
					state["test_gene_sca"]=nn_based_anchor_detection(ref_adata_sca=state["ref_gene_sca"], test_adata_sca=state["test_gene_sca"], combined_genes=target_genes+nontgt_genes, target_regions=target_regions, nontgt_regions=nontgt_regions, target_node=target_node, nontgt_node=nontgt_node, label_col=label_col, knn=knn, metric=metric)
		return state
	elif anchor_mode=="range":
		if config.get("reference_mode", "single")=="multi":
			state["test_gene_sca"], _, _=range_based_anchor_detection_upd_v2(ref_adata_sca_dic=state["ref_gene_sca_dic"], merged_ref_adata_sca=state["merged_ref_gene_sca"], test_adata_sca=state["test_gene_sca"], target_node=target_node, nontgt_node=nontgt_node, target_regions=target_regions, nontgt_regions=nontgt_regions, target_genes=target_genes, nontgt_genes=nontgt_genes, perct_cf_upper=config.get("perct_cf_upper", 0.85), perct_cf_lower=config.get("perct_cf_lower", 0.15), max_p_list=config.get("anchor_max_p_list", [0.6, 0.6]), thres_q_list=config.get("anchor_thres_q_list", [0.9, 0.9]), anchor_thres_upper_adjust=config.get("anchor_thres_upper_adjust", False), anchor_thres_lower_adjust=config.get("anchor_thres_lower_adjust", False), label_col=label_col, merged_key=config.get("merged_key", "batch"))
		else:
			state["test_gene_sca"], _, _=range_based_anchor_detection_upd_v2(ref_adata_sca_dic=state["ref_gene_sca_dic"], merged_ref_adata_sca=state["merged_ref_gene_sca"], test_adata_sca=state["test_gene_sca"], target_node=target_node, nontgt_node=nontgt_node, target_regions=target_regions, nontgt_regions=nontgt_regions, target_genes=target_genes, nontgt_genes=nontgt_genes, perct_cf_upper=config.get("perct_cf_upper", 0.85), perct_cf_lower=config.get("perct_cf_lower", 0.15), max_p_list=config.get("anchor_max_p_list", [0.6, 0.6]), thres_q_list=config.get("anchor_thres_q_list", [0.9, 0.9]), anchor_thres_upper_adjust=config.get("anchor_thres_upper_adjust", False), anchor_thres_lower_adjust=config.get("anchor_thres_lower_adjust", False), label_col=label_col, merged_key=config.get("merged_key", "batch"))
		return state
	else:
		raise ValueError(f"Unsupported anchor_mode: {anchor_mode}")


def _assign_labels_state(state, config, target_node, nontgt_node):
	label_mode=config.get("label_mode", "weighted")
	cluster_key=config.get("cluster_key", "leiden_clusters")
	infer_key=target_node+"_vs_"+nontgt_node
	hier_anchor_key=[i+"_anchors" for i in [target_node, nontgt_node]]
	allow_novel_clusters=config.get("allow_novel_clusters", "auto")
	if allow_novel_clusters=="auto":
		allow_novel_clusters=(len(state["target_regions"])==1) or (len(state["nontgt_regions"])==1)
	kwargs=dict(input_adata=state["test_gene_sca"], hier_index=[target_node, nontgt_node], hier_anchor_key=hier_anchor_key, infer_key=infer_key, cluster_key=cluster_key, x_col=config.get("x_col", "x"), y_col=config.get("y_col", "y"), allow_novel_clusters=allow_novel_clusters, drop_thres=config.get("drop_thres", 5), nan_thres=config.get("nan_thres", 0), prop_diff_cutoff=config.get("prop_diff_cutoff", 1), num_nbs=config.get("num_nbs", 15))
	if label_mode=="plain":
		state["test_gene_sca"], _=hier_assign_labels(input_adata=state["test_gene_sca"], hier_index=[target_node, nontgt_node], hier_anchor_key=hier_anchor_key, infer_key=infer_key, cluster_key=cluster_key, drop_thres=config.get("drop_thres", 5), nan_thres=config.get("nan_thres", 0))
	elif label_mode=="weighted_multi":
		state["test_gene_sca"]=hier_assign_labels_weights_multi(**kwargs, anchor_weight_g=config.get("anchor_weight_g", 1), anchor_weight_p=config.get("anchor_weight_p", 1))
	else:
		state["test_gene_sca"]=hier_assign_labels_weights(**kwargs)
	binary_cate=[i for i in state["test_gene_sca"].obs[infer_key].value_counts().index.tolist() if i!="novel_cluster"]
	if len(binary_cate)==1 and config.get("binary_ratio_thres", 5) is not None:
		ref_ratio=1.0
		if "ref_gene_sca" in state and state["ref_gene_sca"] is not None:
			ref_counts=state["ref_gene_sca"].obs[config.get("label_col", "label")].value_counts()
			target_spots_num=ref_counts.reindex(state["target_regions"], fill_value=0).sum()
			nontgt_spots_num=ref_counts.reindex(state["nontgt_regions"], fill_value=0).sum()
			ref_ratio=np.max([target_spots_num, nontgt_spots_num])/max(1, np.min([target_spots_num, nontgt_spots_num]))
		elif "ref_gene_sca_dic" in state and isinstance(state["ref_gene_sca_dic"], dict):
			ref_counts_list=[]
			for ref_adata in state["ref_gene_sca_dic"].values():
				ref_counts_list.append(ref_adata.obs[config.get("label_col", "label")].value_counts())
			if len(ref_counts_list)>0:
				ref_counts=pd.concat(ref_counts_list, axis=1).fillna(0).sum(axis=1)
				target_spots_num=ref_counts.reindex(state["target_regions"], fill_value=0).sum()
				nontgt_spots_num=ref_counts.reindex(state["nontgt_regions"], fill_value=0).sum()
				ref_ratio=np.max([target_spots_num, nontgt_spots_num])/max(1, np.min([target_spots_num, nontgt_spots_num]))
		if ref_ratio>config.get("binary_ratio_thres", 5):
			print("**************************** label assignment adjusted by weights ****************************")
			if label_mode=="weighted_multi":
				state["test_gene_sca"]=hier_assign_labels_weights_multi(**kwargs, prop_diff_cutoff=config.get("prop_diff_cutoff_upd", 100), anchor_weight_g=config.get("anchor_weight_g", 1), anchor_weight_p=config.get("anchor_weight_p", 1))
			else:
				state["test_gene_sca"]=hier_assign_labels_weights(**kwargs, prop_diff_cutoff=config.get("prop_diff_cutoff_upd", 100))
	_state_label=state["test_gene_sca"].obs[infer_key].copy()
	_state_label=_state_label.astype("category")
	state["test_gene_sca"].obs[infer_key]=_state_label
	return state, infer_key


def _run_recursive_binary_transfer(state, config, qry_nodes_dic=None):
	if qry_nodes_dic is None:
		qry_nodes_dic={}
	root_node=state["root_node"]
	if root_node in qry_nodes_dic and config.get("stop_if_seen_root", False):
		return qry_nodes_dic
	if root_node in state["region_nodes"]:
		return qry_nodes_dic
	if root_node not in state["adj_dic"]:
		return qry_nodes_dic
	split_nodes=state["adj_dic"][root_node]
	target_node, nontgt_node=split_nodes[0], split_nodes[1]
	state["target_regions"]=state["node_dic"][target_node]
	state["nontgt_regions"]=state["node_dic"][nontgt_node]
	if config.get("reference_mode", "single")=="multi":
		target_genes, nontgt_genes=_collect_edge_features_from_dicts(state["gene_feature_dicts"], target_node, nontgt_node, sections_used=state.get("sections_used"))
		target_image, nontgt_image=_collect_edge_features_from_dicts(state.get("image_feature_dicts", []), target_node, nontgt_node, sections_used=state.get("sections_used"))
		target_protein, nontgt_protein=_collect_edge_features_from_dicts(state.get("protein_feature_dicts", []), target_node, nontgt_node, sections_used=state.get("sections_used"))
	else:
		target_genes=state["hier_genes_dic"][target_node+"_vs_"+nontgt_node]
		nontgt_genes=state["hier_genes_dic"][nontgt_node+"_vs_"+target_node]
		target_image=state.get("hier_image_dic", {}).get(target_node+"_vs_"+nontgt_node, [])
		nontgt_image=state.get("hier_image_dic", {}).get(nontgt_node+"_vs_"+target_node, [])
		target_protein=state.get("hier_proteins_dic", {}).get(target_node+"_vs_"+nontgt_node, [])
		nontgt_protein=state.get("hier_proteins_dic", {}).get(nontgt_node+"_vs_"+target_node, [])
	state, cluster_key=_cluster_state(state, config, target_node, nontgt_node, target_genes, nontgt_genes, target_image=target_image, nontgt_image=nontgt_image, target_protein=target_protein, nontgt_protein=nontgt_protein)
	state["cluster_key"]=cluster_key
	config["cluster_key"]=cluster_key
	state=_detect_anchors_state(state, config, target_node, nontgt_node, state["target_regions"], state["nontgt_regions"], target_genes, nontgt_genes, target_protein=target_protein, nontgt_protein=nontgt_protein)
	state, infer_key=_assign_labels_state(state, config, target_node, nontgt_node)
	labels=state["test_gene_sca"].obs[infer_key].copy()
	if "test_hipt" in state and state["test_hipt"] is not None:
		state["test_hipt"].obs[infer_key]=labels.copy()
		state["test_hipt"].obs[infer_key]=state["test_hipt"].obs[infer_key].astype("category")
	if "test_gene" in state and state["test_gene"] is not None:
		state["test_gene"].obs[infer_key]=labels.copy()
		state["test_gene"].obs[infer_key]=state["test_gene"].obs[infer_key].astype("category")
	if "test_protein" in state and state["test_protein"] is not None:
		state["test_protein"].obs[infer_key]=labels.copy()
		state["test_protein"].obs[infer_key]=state["test_protein"].obs[infer_key].astype("category")
	if "test_protein_sca" in state and state["test_protein_sca"] is not None:
		state["test_protein_sca"].obs[infer_key]=labels.copy()
		state["test_protein_sca"].obs[infer_key]=state["test_protein_sca"].obs[infer_key].astype("category")
	fig_title=root_node+"_"+target_node+"+"+nontgt_node+": binary separation"
	fig_path=config["plot_dir"]+"/"+root_node+"_"+target_node+"+"+nontgt_node+"_binary_separation.png"
	cat_figure(input_adata=state["test_gene_sca"], x_col=config.get("x_col", "x"), y_col=config.get("y_col", "y"), fig_title=fig_title, fig_path=fig_path, color_key=infer_key, cat_color=cat_color, size=config.get("fig_size", 15), invert_x=False, invert_y=True)
	qry_nodes_dic[root_node]=state["test_gene_sca"].obs.index.tolist()
	for node in split_nodes:
		node_prop=state["test_gene_sca"].obs[infer_key].value_counts(normalize=True)
		if node in node_prop.index.tolist():
			qry_nodes_dic[node]=state["test_gene_sca"][state["test_gene_sca"].obs[infer_key]==node].obs.index.tolist()
			if config.get("recursive", True) and (node not in state["region_nodes"]) and (node_prop[node]>=config.get("min_cluster_fraction", 0.05)):
				child_state=state.copy()
				child_state["root_node"]=node
				child_state["test_gene_sca"]=state["test_gene_sca"][state["test_gene_sca"].obs[infer_key]==node].copy()
				if child_state.get("test_gene") is not None:
					child_state["test_gene"]=state["test_gene"][state["test_gene"].obs[infer_key]==node].copy()
				if child_state.get("test_hipt") is not None:
					child_state["test_hipt"]=state["test_hipt"][state["test_hipt"].obs[infer_key]==node].copy()
				if child_state.get("test_protein") is not None:
					child_state["test_protein"]=state["test_protein"][state["test_protein"].obs[infer_key]==node].copy()
				if child_state.get("test_protein_sca") is not None:
					child_state["test_protein_sca"]=state["test_protein_sca"][state["test_protein_sca"].obs[infer_key]==node].copy()
				qry_nodes_dic=_run_recursive_binary_transfer(child_state, config, qry_nodes_dic=qry_nodes_dic)
	return qry_nodes_dic


def _build_state_single_ref(ref_gene_sca, test_gene_sca, test_hipt=None, test_gene=None, test_protein_sca=None, test_protein=None, root_node=None, region_nodes=None, hier_genes_dic=None, hier_image_dic=None, hier_proteins_dic=None, adj_dic=None, node_dic=None, plot_dir=None, cluster_key="leiden_clusters", label_col="label", ref_spots_filtering=False, **extra):
	state=dict(
		reference_mode="single",
		root_node=root_node,
		region_nodes=region_nodes,
		ref_gene_sca=ref_gene_sca,
		test_gene_sca=test_gene_sca,
		test_hipt=test_hipt,
		test_gene=test_gene,
		test_protein_sca=test_protein_sca,
		test_protein=test_protein,
		hier_genes_dic=hier_genes_dic or {},
		hier_image_dic=hier_image_dic or {},
		hier_proteins_dic=hier_proteins_dic or {},
		adj_dic=adj_dic or {},
		node_dic=node_dic or {},
		plot_dir=plot_dir,
		cluster_key=cluster_key,
		label_col=label_col,
		ref_spots_filtering=ref_spots_filtering,
	)
	state.update(extra)
	return state


def _build_state_multi_ref(ref_gene_sca_list, ref_names, test_gene_sca, test_hipt=None, test_gene=None, test_protein_sca=None, test_protein=None, root_node=None, region_nodes=None, gene_feature_dicts=None, image_feature_dicts=None, protein_feature_dicts=None, merged_ref_gene_sca=None, adj_dic=None, node_dic=None, hier_sec_used=None, plot_dir=None, cluster_key="leiden_clusters", label_col="label", ref_spots_filtering=False, **extra):
	ref_payloads=[{"name": name, "gene": ref} for name, ref in zip(ref_names, ref_gene_sca_list)]
	state=dict(
		reference_mode="multi",
		root_node=root_node,
		region_nodes=region_nodes,
		ref_payloads=ref_payloads,
		ref_gene_sca_dic={name: ref for name, ref in zip(ref_names, ref_gene_sca_list)},
		merged_ref_gene_sca=merged_ref_gene_sca,
		test_gene_sca=test_gene_sca,
		test_hipt=test_hipt,
		test_gene=test_gene,
		test_protein_sca=test_protein_sca,
		test_protein=test_protein,
		gene_feature_dicts=gene_feature_dicts or [],
		image_feature_dicts=image_feature_dicts or [],
		protein_feature_dicts=protein_feature_dicts or [],
		adj_dic=adj_dic or {},
		node_dic=node_dic or {},
		hier_sec_used=hier_sec_used or {},
		sections_used=hier_sec_used.get(root_node, []) if isinstance(hier_sec_used, dict) else [],
		plot_dir=plot_dir,
		cluster_key=cluster_key,
		label_col=label_col,
		ref_spots_filtering=ref_spots_filtering,
	)
	state.update(extra)
	return state


def _run_pipeline_with_config(state, config, qry_nodes_dic=None):
	if qry_nodes_dic is None:
		qry_nodes_dic={}
	config=dict(config)
	config.setdefault("plot_dir", state["plot_dir"])
	config.setdefault("cluster_key", state.get("cluster_key", "leiden_clusters"))
	config.setdefault("label_col", state.get("label_col", "label"))
	config.setdefault("x_col", state.get("x_col", "x"))
	config.setdefault("y_col", state.get("y_col", "y"))
	state["plot_dir"]=config["plot_dir"]
	state["cluster_key"]=config["cluster_key"]
	state["label_col"]=config["label_col"]
	return _run_recursive_binary_transfer(state, config, qry_nodes_dic=qry_nodes_dic)


# ----------------------- preset wrappers -----------------------
def single_ref_I_nn_based_hier_clustering(ref_gene_sca, test_gene_sca, test_hipt, root_node, region_nodes, hier_genes_dic, adj_dic, node_dic, plot_dir, hipt_npcs=10, leiden_res=0.2, n_neighbors=10, num_nbs=15, knn=5, binary_ratio_thres=5, drop_thres=5, nan_thres=0, prop_diff_cutoff=1, fig_size=15, x_col="y", y_col="x", label_col="label", cluster_key="leiden_clusters", ref_spots_filtering=False, qry_nodes_dic=None):
	state=_build_state_single_ref(ref_gene_sca=ref_gene_sca, test_gene_sca=test_gene_sca, test_hipt=test_hipt, root_node=root_node, region_nodes=region_nodes, hier_genes_dic=hier_genes_dic, adj_dic=adj_dic, node_dic=node_dic, plot_dir=plot_dir, cluster_key=cluster_key, label_col=label_col, ref_spots_filtering=ref_spots_filtering, x_col=x_col, y_col=y_col)
	config=dict(reference_mode="single", modality_mode="image_gene", cluster_source="hipt", cluster_backend="leiden", cluster_outputs=["test_gene_sca", "test_hipt"], feature_transform="pca", resolution=leiden_res, n_neighbors=n_neighbors, hipt_npcs=hipt_npcs, anchor_mode="nn", label_mode="weighted", recursive=True, num_nbs=num_nbs, knn=knn, binary_ratio_thres=binary_ratio_thres, drop_thres=drop_thres, nan_thres=nan_thres, prop_diff_cutoff=prop_diff_cutoff, fig_size=fig_size, x_col=x_col, y_col=y_col, label_col=label_col, ref_spots_filtering=ref_spots_filtering)
	return _run_pipeline_with_config(state, config, qry_nodes_dic=qry_nodes_dic)


def single_ref_G_nn_based_hier_clustering(ref_gene_sca, test_gene_sca, test_gene, root_node, region_nodes, hier_genes_dic, adj_dic, node_dic, plot_dir, leiden_res=0.2, n_neighbors=10, bd_num_nbs=5, knn=5, binary_ratio_thres=5, drop_thres=5, nan_thres=0, prop_diff_cutoff=1, fig_size=15, x_col="y", y_col="x", label_col="label", cluster_key="leiden_clusters", ref_spots_filtering=False, qry_nodes_dic=None):
	state=_build_state_single_ref(ref_gene_sca=ref_gene_sca, test_gene_sca=test_gene_sca, test_gene=test_gene, root_node=root_node, region_nodes=region_nodes, hier_genes_dic=hier_genes_dic, adj_dic=adj_dic, node_dic=node_dic, plot_dir=plot_dir, cluster_key=cluster_key, label_col=label_col, ref_spots_filtering=ref_spots_filtering, x_col=x_col, y_col=y_col)
	config=dict(reference_mode="single", modality_mode="gene", cluster_source="gene", cluster_backend="leiden", cluster_outputs=["test_gene_sca", "test_gene"], feature_transform="raw", resolution=leiden_res, n_neighbors=n_neighbors, anchor_mode="nn", label_mode="weighted", recursive=True, num_nbs=bd_num_nbs, knn=knn, binary_ratio_thres=binary_ratio_thres, drop_thres=drop_thres, nan_thres=nan_thres, prop_diff_cutoff=prop_diff_cutoff, fig_size=fig_size, x_col=x_col, y_col=y_col, label_col=label_col, ref_spots_filtering=ref_spots_filtering)
	return _run_pipeline_with_config(state, config, qry_nodes_dic=qry_nodes_dic)


def single_ref_G_PCA_nn_based_hier_clustering(ref_gene_sca, test_gene_sca, test_gene, root_node, region_nodes, hier_genes_dic, adj_dic, node_dic, plot_dir, gene_npcs=30, leiden_res=0.2, n_neighbors=10, bd_num_nbs=5, knn=5, binary_ratio_thres=5, drop_thres=5, nan_thres=0, prop_diff_cutoff=1, fig_size=15, x_col="y", y_col="x", label_col="label", cluster_key="leiden_clusters", ref_spots_filtering=False, qry_nodes_dic=None):
	state=_build_state_single_ref(ref_gene_sca=ref_gene_sca, test_gene_sca=test_gene_sca, test_gene=test_gene, root_node=root_node, region_nodes=region_nodes, hier_genes_dic=hier_genes_dic, adj_dic=adj_dic, node_dic=node_dic, plot_dir=plot_dir, cluster_key=cluster_key, label_col=label_col, ref_spots_filtering=ref_spots_filtering, x_col=x_col, y_col=y_col)
	config=dict(reference_mode="single", modality_mode="gene", cluster_source="gene", cluster_backend="leiden", cluster_outputs=["test_gene_sca", "test_gene"], feature_transform="pca", gene_npcs=gene_npcs, resolution=leiden_res, n_neighbors=n_neighbors, anchor_mode="nn", label_mode="weighted", recursive=True, num_nbs=bd_num_nbs, knn=knn, binary_ratio_thres=binary_ratio_thres, drop_thres=drop_thres, nan_thres=nan_thres, prop_diff_cutoff=prop_diff_cutoff, fig_size=fig_size, x_col=x_col, y_col=y_col, label_col=label_col, ref_spots_filtering=ref_spots_filtering)
	return _run_pipeline_with_config(state, config, qry_nodes_dic=qry_nodes_dic)


def single_ref_I_G_nn_based_hier_clustering_v3(ref_gene_sca, test_gene_sca, test_gene, test_hipt, root_node, region_nodes, hier_genes_dic, hier_image_dic, adj_dic, node_dic, plot_dir, leiden_res=0.2, n_neighbors=10, num_nbs=10, knn=5, binary_ratio_thres=5, drop_thres=5, nan_thres=0, prop_diff_cutoff=1, fig_size=15, x_col="y", y_col="x", label_col="label", cluster_key="leiden_clusters", ref_spots_filtering=False, qry_nodes_dic=None):
	state=_build_state_single_ref(ref_gene_sca=ref_gene_sca, test_gene_sca=test_gene_sca, test_hipt=test_hipt, test_gene=test_gene, root_node=root_node, region_nodes=region_nodes, hier_genes_dic=hier_genes_dic, hier_image_dic=hier_image_dic, adj_dic=adj_dic, node_dic=node_dic, plot_dir=plot_dir, cluster_key=cluster_key, label_col=label_col, ref_spots_filtering=ref_spots_filtering, x_col=x_col, y_col=y_col)
	config=dict(reference_mode="single", modality_mode="gene_image", cluster_source="combined", cluster_backend="leiden", cluster_outputs=["test_gene_sca", "test_hipt", "test_gene"], feature_transform="raw", resolution=leiden_res, n_neighbors=n_neighbors, anchor_mode="nn", label_mode="weighted", recursive=True, num_nbs=num_nbs, knn=knn, binary_ratio_thres=binary_ratio_thres, drop_thres=drop_thres, nan_thres=nan_thres, prop_diff_cutoff=prop_diff_cutoff, fig_size=fig_size, x_col=x_col, y_col=y_col, label_col=label_col, ref_spots_filtering=ref_spots_filtering)
	return _run_pipeline_with_config(state, config, qry_nodes_dic=qry_nodes_dic)


def single_ref_I_G_P_nn_based_hier_clustering_v1(ref_gene_sca, ref_protein_sca, test_gene_sca, test_gene, test_protein_sca, test_protein, test_hipt, root_node, region_nodes, hier_genes_dic, hier_image_dic, hier_proteins_dic, adj_dic, node_dic, plot_dir, leiden_res=0.2, n_neighbors=10, num_nbs=10, knn=5, anchor_weight_g=1, anchor_weight_p=1, binary_ratio_thres=5, drop_thres=5, nan_thres=0, prop_diff_cutoff=1, fig_size=15, x_col="y", y_col="x", label_col="label", cluster_key="leiden_clusters", ref_spots_filtering=False, qry_nodes_dic=None):
	state=_build_state_single_ref(ref_gene_sca=ref_gene_sca, test_gene_sca=test_gene_sca, test_hipt=test_hipt, test_gene=test_gene, test_protein_sca=test_protein_sca, test_protein=test_protein, root_node=root_node, region_nodes=region_nodes, hier_genes_dic=hier_genes_dic, hier_image_dic=hier_image_dic, hier_proteins_dic=hier_proteins_dic, adj_dic=adj_dic, node_dic=node_dic, plot_dir=plot_dir, cluster_key=cluster_key, label_col=label_col, ref_spots_filtering=ref_spots_filtering, x_col=x_col, y_col=y_col, ref_protein_sca=ref_protein_sca)
	config=dict(reference_mode="single", modality_mode="gene_image_protein", cluster_source="combined", cluster_backend="leiden", cluster_outputs=["test_gene_sca", "test_hipt", "test_gene", "test_protein_sca", "test_protein"], feature_transform="raw", resolution=leiden_res, n_neighbors=n_neighbors, anchor_mode="nn", label_mode="weighted_multi", recursive=True, num_nbs=num_nbs, knn=knn, anchor_weight_g=anchor_weight_g, anchor_weight_p=anchor_weight_p, binary_ratio_thres=binary_ratio_thres, drop_thres=drop_thres, nan_thres=nan_thres, prop_diff_cutoff=prop_diff_cutoff, fig_size=fig_size, x_col=x_col, y_col=y_col, label_col=label_col, ref_spots_filtering=ref_spots_filtering)
	return _run_pipeline_with_config(state, config, qry_nodes_dic=qry_nodes_dic)


def one_round_I_G_range_based_binary_separation_upd_v2(ref_gene_sca_dic, merged_ref_gene_sca, test_gene_sca, test_gene, test_hipt, root_node, region_nodes, sudo_hier_genes_dic, adj_dic, node_dic, plot_dir, hipt_npcs=10, leiden_res=0.2, n_neighbors=10, n_clusters=5, bd_num_nbs_2=15, perct_cf_upper=0.85, perct_cf_lower=0.15, anchor_thres_q_list=[0.9, 0.9], anchor_max_p_list=[0.6, 0.6], drop_thres=5, nan_thres=0, prop_diff_cutoff=1, prop_diff_cutoff_upd=30, fig_size=15, x_col="y", y_col="x", label_col="label", cluster_key="leiden_clusters", merged_key="tissue_section", allow_novel_clusters=True, anchor_thres_upper_adjust=False, anchor_thres_lower_adjust=False, qry_nodes_dic=None):
	state=dict(reference_mode="multi", root_node=root_node, region_nodes=region_nodes, ref_gene_sca_dic=ref_gene_sca_dic, merged_ref_gene_sca=merged_ref_gene_sca, test_gene_sca=test_gene_sca, test_gene=test_gene, test_hipt=test_hipt, adj_dic=adj_dic, node_dic=node_dic, sudo_hier_genes_dic=sudo_hier_genes_dic, plot_dir=plot_dir, cluster_key=cluster_key, label_col=label_col, ref_spots_filtering=False, x_col=x_col, y_col=y_col)
	config=dict(reference_mode="multi", modality_mode="gene_image", cluster_source="hipt", cluster_backend="kmeans" if cluster_key=="kmeans_clusters" else "leiden", cluster_outputs=["test_gene_sca", "test_hipt", "test_gene"], feature_transform="pca", resolution=leiden_res, n_neighbors=n_neighbors, hipt_npcs=hipt_npcs, n_clusters=n_clusters, anchor_mode="range", label_mode="weighted", recursive=False, num_nbs=bd_num_nbs_2, drop_thres=drop_thres, nan_thres=nan_thres, prop_diff_cutoff=prop_diff_cutoff, prop_diff_cutoff_upd=prop_diff_cutoff_upd, fig_size=fig_size, x_col=x_col, y_col=y_col, label_col=label_col, perct_cf_upper=perct_cf_upper, perct_cf_lower=perct_cf_lower, anchor_thres_q_list=anchor_thres_q_list, anchor_max_p_list=anchor_max_p_list, anchor_thres_upper_adjust=anchor_thres_upper_adjust, anchor_thres_lower_adjust=anchor_thres_lower_adjust, merged_key=merged_key, allow_novel_clusters=allow_novel_clusters)
	state["ref_gene_sca_dic"]=ref_gene_sca_dic
	state["sections_used"]=list(ref_gene_sca_dic.keys())
	state["hier_sec_used"]={root_node: list(ref_gene_sca_dic.keys())}
	state["gene_feature_dicts"]=[(k, sudo_hier_genes_dic) for k in ref_gene_sca_dic.keys()]
	return _run_pipeline_with_config(state, config, qry_nodes_dic=qry_nodes_dic)


def two_ref_I_G_nn_based_hier_clustering(ref_gene_sca1, ref_gene_sca2, test_gene_sca, test_gene, test_hipt, ref_section1, ref_section2, root_node, region_nodes, sudo_hier_genes_dic1, sudo_hier_genes_dic2, hier_sec_used, adj_dic, node_dic, plot_dir, hipt_npcs=10, leiden_res=0.2, n_neighbors=10, bd_num_nbs_1=25, bd_num_nbs_2=15, subtype_gene_num=5, subtype_min_prop=0.15, subtype_louvain_res=0.05, subtype_n_neighbors=15, subtype_num_nbs=10, knn=5, drop_thres=5, nan_thres=0, prop_diff_cutoff_upd=20, fig_size=15, x_col="y", y_col="x", label_col="label", cluster_key="leiden_clusters", ref_spots_filtering=False, qry_nodes_dic=None):
	state=_build_state_multi_ref(ref_gene_sca_list=[ref_gene_sca1, ref_gene_sca2], ref_names=[ref_section1, ref_section2], test_gene_sca=test_gene_sca, test_hipt=test_hipt, test_gene=test_gene, root_node=root_node, region_nodes=region_nodes, gene_feature_dicts=[(ref_section1, sudo_hier_genes_dic1), (ref_section2, sudo_hier_genes_dic2)], merged_ref_gene_sca=None, adj_dic=adj_dic, node_dic=node_dic, hier_sec_used=hier_sec_used, plot_dir=plot_dir, cluster_key=cluster_key, label_col=label_col, ref_spots_filtering=ref_spots_filtering, x_col=x_col, y_col=y_col)
	config=dict(reference_mode="multi", modality_mode="gene_image", cluster_source="hipt", cluster_backend="leiden", cluster_outputs=["test_gene_sca", "test_hipt", "test_gene"], feature_transform="pca", resolution=leiden_res, n_neighbors=n_neighbors, hipt_npcs=hipt_npcs, anchor_mode="nn", label_mode="weighted", recursive=True, num_nbs=subtype_num_nbs, knn=knn, binary_ratio_thres=knn, drop_thres=drop_thres, nan_thres=nan_thres, prop_diff_cutoff=0, prop_diff_cutoff_upd=prop_diff_cutoff_upd, fig_size=fig_size, x_col=x_col, y_col=y_col, label_col=label_col, ref_spots_filtering=ref_spots_filtering)
	return _run_pipeline_with_config(state, config, qry_nodes_dic=qry_nodes_dic)


def three_ref_I_G_nn_based_hier_clustering(ref_gene_sca1, ref_gene_sca2, ref_gene_sca3, test_gene_sca, test_gene, test_hipt, ref_section1, ref_section2, ref_section3, root_node, region_nodes, sudo_hier_genes_dic1, sudo_hier_genes_dic2, sudo_hier_genes_dic3, hier_sec_used, adj_dic, node_dic, plot_dir, hipt_npcs=10, leiden_res=0.2, n_neighbors=10, bd_num_nbs_1=25, bd_num_nbs_2=15, subtype_gene_num=5, subtype_min_prop=0.15, subtype_louvain_res=0.05, subtype_n_neighbors=15, subtype_num_nbs=10, knn=5, drop_thres=5, nan_thres=0, prop_diff_cutoff_upd=20, fig_size=15, x_col="y", y_col="x", label_col="label", cluster_key="leiden_clusters", ref_spots_filtering=False, qry_nodes_dic=None):
	state=_build_state_multi_ref(ref_gene_sca_list=[ref_gene_sca1, ref_gene_sca2, ref_gene_sca3], ref_names=[ref_section1, ref_section2, ref_section3], test_gene_sca=test_gene_sca, test_hipt=test_hipt, test_gene=test_gene, root_node=root_node, region_nodes=region_nodes, gene_feature_dicts=[(ref_section1, sudo_hier_genes_dic1), (ref_section2, sudo_hier_genes_dic2), (ref_section3, sudo_hier_genes_dic3)], merged_ref_gene_sca=None, adj_dic=adj_dic, node_dic=node_dic, hier_sec_used=hier_sec_used, plot_dir=plot_dir, cluster_key=cluster_key, label_col=label_col, ref_spots_filtering=ref_spots_filtering, x_col=x_col, y_col=y_col)
	config=dict(reference_mode="multi", modality_mode="gene_image", cluster_source="hipt", cluster_backend="leiden", cluster_outputs=["test_gene_sca", "test_hipt", "test_gene"], feature_transform="pca", resolution=leiden_res, n_neighbors=n_neighbors, hipt_npcs=hipt_npcs, anchor_mode="nn", label_mode="weighted", recursive=True, num_nbs=subtype_num_nbs, knn=knn, binary_ratio_thres=knn, drop_thres=drop_thres, nan_thres=nan_thres, prop_diff_cutoff=0, prop_diff_cutoff_upd=prop_diff_cutoff_upd, fig_size=fig_size, x_col=x_col, y_col=y_col, label_col=label_col, ref_spots_filtering=ref_spots_filtering)
	return _run_pipeline_with_config(state, config, qry_nodes_dic=qry_nodes_dic)


def two_ref_G_PCA_nn_based_hier_clustering(ref_gene_sca1, ref_gene_sca2, test_gene_sca, test_gene, ref_section1, ref_section2, root_node, region_nodes, sudo_hier_genes_dic1, sudo_hier_genes_dic2, hier_sec_used, adj_dic, node_dic, plot_dir, gene_npcs=30, leiden_res=0.2, n_neighbors=10, bd_num_nbs=5, knn=5, drop_thres=5, nan_thres=0, prop_diff_cutoff=5, prop_diff_cutoff_upd=20, fig_size=15, x_col="y", y_col="x", label_col="label", cluster_key="leiden_clusters", ref_spots_filtering=False, qry_nodes_dic=None):
	state=_build_state_multi_ref(ref_gene_sca_list=[ref_gene_sca1, ref_gene_sca2], ref_names=[ref_section1, ref_section2], test_gene_sca=test_gene_sca, test_gene=test_gene, root_node=root_node, region_nodes=region_nodes, gene_feature_dicts=[(ref_section1, sudo_hier_genes_dic1), (ref_section2, sudo_hier_genes_dic2)], merged_ref_gene_sca=None, adj_dic=adj_dic, node_dic=node_dic, hier_sec_used=hier_sec_used, plot_dir=plot_dir, cluster_key=cluster_key, label_col=label_col, ref_spots_filtering=ref_spots_filtering, x_col=x_col, y_col=y_col)
	config=dict(reference_mode="multi", modality_mode="gene", cluster_source="gene", cluster_backend="leiden", cluster_outputs=["test_gene_sca", "test_gene"], feature_transform="pca", gene_npcs=gene_npcs, resolution=leiden_res, n_neighbors=n_neighbors, anchor_mode="nn", label_mode="weighted", recursive=True, num_nbs=bd_num_nbs, knn=knn, binary_ratio_thres=knn, drop_thres=drop_thres, nan_thres=nan_thres, prop_diff_cutoff=prop_diff_cutoff, prop_diff_cutoff_upd=prop_diff_cutoff_upd, fig_size=fig_size, x_col=x_col, y_col=y_col, label_col=label_col, ref_spots_filtering=ref_spots_filtering)
	return _run_pipeline_with_config(state, config, qry_nodes_dic=qry_nodes_dic)


def three_ref_G_PCA_nn_based_hier_clustering(ref_gene_sca1, ref_gene_sca2, ref_gene_sca3, test_gene_sca, test_gene, ref_section1, ref_section2, ref_section3, root_node, region_nodes, sudo_hier_genes_dic1, sudo_hier_genes_dic2, sudo_hier_genes_dic3, hier_sec_used, adj_dic, node_dic, plot_dir, gene_npcs=30, leiden_res=0.2, n_neighbors=10, bd_num_nbs=5, knn=5, drop_thres=5, nan_thres=0, prop_diff_cutoff=5, prop_diff_cutoff_upd=20, fig_size=15, x_col="y", y_col="x", label_col="label", cluster_key="leiden_clusters", ref_spots_filtering=False, qry_nodes_dic=None):
	state=_build_state_multi_ref(ref_gene_sca_list=[ref_gene_sca1, ref_gene_sca2, ref_gene_sca3], ref_names=[ref_section1, ref_section2, ref_section3], test_gene_sca=test_gene_sca, test_gene=test_gene, root_node=root_node, region_nodes=region_nodes, gene_feature_dicts=[(ref_section1, sudo_hier_genes_dic1), (ref_section2, sudo_hier_genes_dic2), (ref_section3, sudo_hier_genes_dic3)], merged_ref_gene_sca=None, adj_dic=adj_dic, node_dic=node_dic, hier_sec_used=hier_sec_used, plot_dir=plot_dir, cluster_key=cluster_key, label_col=label_col, ref_spots_filtering=ref_spots_filtering, x_col=x_col, y_col=y_col)
	config=dict(reference_mode="multi", modality_mode="gene", cluster_source="gene", cluster_backend="leiden", cluster_outputs=["test_gene_sca", "test_gene"], feature_transform="pca", gene_npcs=gene_npcs, resolution=leiden_res, n_neighbors=n_neighbors, anchor_mode="nn", label_mode="weighted", recursive=True, num_nbs=bd_num_nbs, knn=knn, binary_ratio_thres=knn, drop_thres=drop_thres, nan_thres=nan_thres, prop_diff_cutoff=prop_diff_cutoff, prop_diff_cutoff_upd=prop_diff_cutoff_upd, fig_size=fig_size, x_col=x_col, y_col=y_col, label_col=label_col, ref_spots_filtering=ref_spots_filtering)
	return _run_pipeline_with_config(state, config, qry_nodes_dic=qry_nodes_dic)


def multi_ref_I_range_based_hier_clustering(ref_gene_sca_dic, merged_ref_gene_sca, test_gene_sca, test_gene, test_hipt, root_node, region_nodes, sudo_hier_genes_dic, adj_dic, node_dic, plot_dir, hipt_npcs=10, leiden_res=0.2, n_neighbors=10, bd_num_nbs_1=25, bd_num_nbs_2=15, perct_cf_upper=0.85, perct_cf_lower=0.15, anchor_thres_q=0.9, anchor_max_p=0.6, drop_thres=5, nan_thres=0, prop_diff_cutoff=1, prop_diff_cutoff_upd=30, fig_size=15, x_col="y", y_col="x", label_col="label", cluster_key="leiden_clusters", merged_key="tissue_section", anchor_thres_upper_adjust=False, anchor_thres_lower_adjust=False, qry_nodes_dic=None):
	state=dict(reference_mode="multi", root_node=root_node, region_nodes=region_nodes, ref_gene_sca_dic=ref_gene_sca_dic, merged_ref_gene_sca=merged_ref_gene_sca, test_gene_sca=test_gene_sca, test_gene=test_gene, test_hipt=test_hipt, adj_dic=adj_dic, node_dic=node_dic, sudo_hier_genes_dic=sudo_hier_genes_dic, plot_dir=plot_dir, cluster_key=cluster_key, label_col=label_col, ref_spots_filtering=False, x_col=x_col, y_col=y_col, gene_feature_dicts=[(k, sudo_hier_genes_dic) for k in ref_gene_sca_dic.keys()], sections_used=list(ref_gene_sca_dic.keys()))
	config=dict(reference_mode="multi", modality_mode="gene_image", cluster_source="hipt", cluster_backend="leiden", cluster_outputs=["test_gene_sca", "test_hipt", "test_gene"], feature_transform="pca", resolution=leiden_res, n_neighbors=n_neighbors, hipt_npcs=hipt_npcs, anchor_mode="range", label_mode="weighted", recursive=True, num_nbs=bd_num_nbs_2, drop_thres=drop_thres, nan_thres=nan_thres, prop_diff_cutoff=prop_diff_cutoff, prop_diff_cutoff_upd=prop_diff_cutoff_upd, fig_size=fig_size, x_col=x_col, y_col=y_col, label_col=label_col, perct_cf_upper=perct_cf_upper, perct_cf_lower=perct_cf_lower, anchor_thres_q_list=[anchor_thres_q, anchor_thres_q], anchor_max_p_list=[anchor_max_p, anchor_max_p], anchor_thres_upper_adjust=anchor_thres_upper_adjust, anchor_thres_lower_adjust=anchor_thres_lower_adjust, merged_key=merged_key)
	return _run_pipeline_with_config(state, config, qry_nodes_dic=qry_nodes_dic)


def multi_ref_G_range_based_hier_clustering(ref_gene_sca_dic, merged_ref_gene_sca, test_gene_sca, test_gene, root_node, region_nodes, hier_genes_dic, adj_dic, node_dic, plot_dir, gene_npcs=30, leiden_res=0.2, n_neighbors=10, bd_num_nbs=5, perct_cf_upper=0.85, perct_cf_lower=0.15, anchor_thres_q=0.9, anchor_max_p=0.6, drop_thres=5, nan_thres=0, prop_diff_cutoff=1, prop_diff_cutoff_upd=30, fig_size=15, x_col="y", y_col="x", label_col="label", cluster_key="leiden_clusters", merged_key="tissue_section", anchor_thres_upper_adjust=False, anchor_thres_lower_adjust=False, qry_nodes_dic=None):
	state=dict(reference_mode="multi", root_node=root_node, region_nodes=region_nodes, ref_gene_sca_dic=ref_gene_sca_dic, merged_ref_gene_sca=merged_ref_gene_sca, test_gene_sca=test_gene_sca, test_gene=test_gene, adj_dic=adj_dic, node_dic=node_dic, hier_genes_dic=hier_genes_dic, plot_dir=plot_dir, cluster_key=cluster_key, label_col=label_col, ref_spots_filtering=False, x_col=x_col, y_col=y_col, gene_feature_dicts=[(k, hier_genes_dic) for k in ref_gene_sca_dic.keys()], sections_used=list(ref_gene_sca_dic.keys()))
	config=dict(reference_mode="multi", modality_mode="gene", cluster_source="gene", cluster_backend="leiden", cluster_outputs=["test_gene_sca", "test_gene"], feature_transform="pca", gene_npcs=gene_npcs, resolution=leiden_res, n_neighbors=n_neighbors, anchor_mode="range", label_mode="weighted", recursive=True, num_nbs=bd_num_nbs, drop_thres=drop_thres, nan_thres=nan_thres, prop_diff_cutoff=prop_diff_cutoff, prop_diff_cutoff_upd=prop_diff_cutoff_upd, fig_size=fig_size, x_col=x_col, y_col=y_col, label_col=label_col, perct_cf_upper=perct_cf_upper, perct_cf_lower=perct_cf_lower, anchor_thres_q_list=[anchor_thres_q, anchor_thres_q], anchor_max_p_list=[anchor_max_p, anchor_max_p], anchor_thres_upper_adjust=anchor_thres_upper_adjust, anchor_thres_lower_adjust=anchor_thres_lower_adjust, merged_key=merged_key)
	return _run_pipeline_with_config(state, config, qry_nodes_dic=qry_nodes_dic)



