import os,csv,sys,time
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sn
import anndata as ad
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import seaborn as sns 
import scanpy.external as sce
import cv2
import math
import time
import hnswlib
from anndata import AnnData
from scipy.sparse import issparse
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from contextlib import redirect_stdout
import matplotlib.patches as mpatches


def section_subtype_DE_genes(adata_dic,tissue_section_list,target_region,res_dir,pcs_num=30,cluster_method="leiden_clusters",n_clusters=2,leiden_res=0.5,n_neighbors=15,random_state=0,cat_color,x_col="pixel_x",y_col="pixel_y",fig_scale=2500,invert_x=False,invert_y=False,pvals_adj=0.05,min_in_out_group_ratio=1,min_in_group_fraction=0.5,min_fold_change=1.1,gene_num=10, cnt_colormap="coolwarm"):
	d_g={}
	total_clusters_num=0
	for tissue_section in tissue_section_list:
		print("======================================= "+tissue_section+" =======================================")
		test_gene=adata_dic[tissue_section].copy()
		#=========================================================================================
		#####------------ Step 1. Identify subclusters within each tissue section -----------#####
		#=========================================================================================
		# reduce dimension
		#---------------- 1. PCA ----------------
		pca=PCA(n_components=pcs_num, random_state=random_state)
		gene_pcs=pca.fit_transform(test_gene.X)
		# clustering
		if cluster_method=="kmeans_clusters":
			#---------------- 2.1 kmeans clustering ----------------
			cluster_key="kmeans_clusters"
			pred=kmeans_clustering(features_matrix=gene_pcs, n_clusters=n_clusters, random_state=random_state, kmeans_key=cluster_key)
			fig_path=res_dir+"/"+tissue_section+"_"+target_region+"_subtype_"+cluster_key+"_npcs="+str(pcs_num)+"_nclusters="+str(n_clusters)+".png"
		elif cluster_method=="leiden_clusters":
			#---------------- 2.2 leiden clustering ----------------
			cluster_key="leiden_clusters"
			pred=leiden_clustering(features_matrix=gene_pcs, resolution=leiden_res, n_neighbors=n_neighbors, random_state=random_state, leiden_key=cluster_key)
			fig_path=res_dir+"/"+tissue_section+"_"+target_region+"_subtype_"+cluster_key+"_npcs="+str(pcs_num)+"_res="+str(leiden_res)+"_nn="+str(n_neighbors)+".png"
		test_gene.obs[cluster_key]=pred.copy()
		test_gene.obs[cluster_key]=test_gene.obs[cluster_key].astype("category")
		cluster_perct=test_gene.obs[cluster_key].value_counts(normalize=True)
		print(cluster_perct)
		# check clustering patterns
		fig_title=f"{tissue_section}: {target_region} subtypes ({cluster_key})"
		os.makedirs(os.path.dirname(fig_path), exist_ok=True)
		cat_figure(input_adata=test_gene, x_col=x_col, y_col=y_col, fig_title=fig_title, fig_path=fig_path, color_key=cluster_key, cat_color=cat_color, size=fig_scale/(test_gene.shape[0]**0.5), invert_x=invert_x, invert_y=invert_y)
		#=========================================================================================
		#####------------ Step 2. Identify region-specific marker genes -----------#####
		#=========================================================================================
		# One vs. All to select genes
		d_g_section={}
		#cluster_list=np.unique(pred).tolist()
		cluster_list=cluster_perct[cluster_perct>0.05].index.tolist()
		total_clusters_num=total_clusters_num+len(cluster_list)
		print("The total number subtype clusters across tissue section is "+str(total_clusters_num))
		if len(cluster_list)>1:
			for target_cluster in cluster_list:
				test_gene.obs["target"]=(test_gene.obs[cluster_key]==target_cluster)*1
				df1=rank_genes_groups(input_adata=test_gene, target=1, label_col="target", non_target="rest", two_sides=False, logged=True)
				# filter genes based on filtering parameters
				#---------------------- df1 ----------------------
				df1_filtered=df1[(df1["pvals_adj"]<=pvals_adj) &
							 	 (df1["in_out_group_ratio"]>=min_in_out_group_ratio) &
							 	 (df1["in_group_fraction"]>=min_in_group_fraction) &
							     (df1["fold_change"]>=min_fold_change)]
				df1_filtered=df1_filtered.sort_values(by="fold_change", ascending=False)
				print("----------------------- after filtering parameters -----------------------")
				print(df1_filtered)	
				filtered_ngene=df1_filtered.shape[0]
				df1_genes=df1_filtered["genes"].tolist()[0:np.min([gene_num, filtered_ngene])]
				print(df1_filtered.iloc[0:gene_num])
				print(df1_genes)		
				d_g_section[f"cluster{target_cluster}_vs_others"]=df1_genes
				# check gene exp patterns
				for g in df1_genes:
					test_gene.obs[g]=test_gene[:,g].X.toarray().flatten()
					fig_title=f"{tissue_section}: cluster{target_cluster} ({g})"
					fig_path=f"{res_dir}/subtype_DE_patterns/{tissue_section}/{tissue_section}_cluster{target_cluster}_{g}.png"
					os.makedirs(os.path.dirname(fig_path), exist_ok=True)
					con_figure(input_adata=test_gene, x_col=x_col, y_col=y_col, fig_title=fig_title, fig_path=fig_path, color_key=g, cnt_color=cnt_colormap, size=fig_scale/(test_gene.shape[0]**0.5), invert_x=invert_x, invert_y=invert_y)
		# summarize genes set
		section_gene_list=[]
		for _, genes in d_g_section.items():
			section_gene_list=list(set(section_gene_list) | set(genes))
		d_g[tissue_section]=section_gene_list
	print(d_g)
	total_genes_list=[]
	for _, gene_list in d_g.items():
		total_genes_list=list(set(total_genes_list) | set(gene_list))
	print(f"The total number of genes is: {len(total_genes_list)}")
	print(total_genes_list)
	return d_g, total_genes_list, total_clusters_num


def identify_shared_subtype(adata_dic,tissue_section_list,target_region,total_genes_list,total_clusters_num,res_dir,cat_color,x_col="pixel_x",y_col="pixel_y",fig_scale=2500,invert_x=False,invert_y=False,set_clusters_num=None,merged_key="sample",random_state=0):
	#=========================================================================================
	#####------------ Step 1. Construct merged adata -----------#####
	#=========================================================================================
	adata_sca_list=[]
	for tissue_section in tissue_section_list:
		test_gene=adata_dic[tissue_section]
		# reduce dimensions by total_genes_list (obtained from Part 1. the union set of section-specific subtype marker genes)
		test_gene_sub=test_gene[:,test_gene.var.index.isin(total_genes_list)].copy()
		# normalization
		test_gene_sub_sca=normalize_adata(test_gene_sub)
		adata_sca_list.append(test_gene_sub_sca)
	merged_adata_sca=ad.concat(adata_sca_list, axis=0, join="inner", label=merged_key, keys=tissue_section_list)
	merged_adata_sca.var["genes"]=merged_adata_sca.var.index.tolist()
	print(merged_adata_sca.var)
	print(merged_adata_sca.obs[merged_key].value_counts())
	#=========================================================================================
	#####------------ Step 2. Identify shared clusters across sections -----------#####
	#=========================================================================================
	# 1. determine the number of clusters
	gene_df=merged_adata_sca.X.copy()
	gene_df=pd.DataFrame(gene_df, index=merged_adata_sca.obs.index, columns=merged_adata_sca.var_names)
	# test different cluster numbers
	all_scores={}
	best_score=-1
	best_k=None
	for k in range(2, total_clusters_num+1):
		kmeans=KMeans(n_clusters=k, random_state=0)
		labels=kmeans.fit_predict(gene_df)
		score=silhouette_score(gene_df, labels)
		print(f"Clusters: {k}, Silhouette Score: {score}")
		all_scores[k]=score
		if score>best_score:
			best_score=score
			best_k=k
	print(f"\nBest number of clusters (by sihouette): {best_k}, score={best_score}") # identify the best number of clusters by the largest silhouette score
	if set_clusters_num!=None:
		best_k=set_clusters_num
		print("Update the number of clusters by the specified number of clusters")
	# 2. clustering
	#------------------------- kmeans clustering -------------------------
	cluster_key="kmeans_clusters"
	pred=kmeans_clustering(features_matrix=merged_adata_sca.X, n_clusters=best_k, random_state=random_state, kmeans_key=cluster_key)
	merged_adata_sca.obs[cluster_key]=pred.copy()
	merged_adata_sca.obs[cluster_key]=merged_adata_sca.obs[cluster_key].astype("category")
	print(merged_adata_sca.obs[cluster_key].value_counts(normalize=True))
	# check patterns
	for tissue_section in tissue_section_list:
		section_adata=merged_adata_sca[merged_adata_sca.obs[merged_key]==tissue_section].copy()
		section_clusters=section_adata.obs[cluster_key].value_counts().index.tolist()
		section_cat_colors=[cat_color[i] for i in sorted(section_clusters)]
		fig_title=tissue_section+": kmeans clustering (subtype cluster num="+str(best_k)+")"
		fig_path=res_dir+"/"+tissue_section+"_"+cluster_key+"_nclusters="+str(best_k)+".png"
		os.makedirs(os.path.dirname(fig_path), exist_ok=True)
		cat_figure(input_adata=section_adata, x_col=x_col, y_col=y_col, fig_title=fig_title, fig_path=fig_path, color_key=cluster_key, cat_color=section_cat_colors, size=fig_scale/(section_adata.shape[0]**0.5), invert_x=invert_x, invert_y=invert_y)
	#------------------------- heatmap clustering -------------------------
	cluster_key="heatmap_clusters"
	# check heatmap patterns
	fig_path=res_dir+"/"+cluster_key+".png"
	os.makedirs(os.path.dirname(fig_path), exist_ok=True)
	cluster_grid=sns.clustermap(gene_df, cmap="coolwarm", method="ward", figsize=(10,10))
	cluster_grid.figure.savefig(fig_path)
	plt.clf()
	plt.close()
	# heatmap clustering
	spots_linkage=cluster_grid.dendrogram_row.linkage
	genes_linkage=cluster_grid.dendrogram_col.linkage
	spots_clusters=fcluster(spots_linkage, best_k, criterion='maxclust')
	genes_clusters=fcluster(genes_linkage, best_k, criterion='maxclust')
	spots_cluster_labels=pd.Series(spots_clusters, index=gene_df.index, name="Spots_clusters")
	genes_cluster_labels=pd.Series(genes_clusters, index=gene_df.columns, name="Gene_clusters")
	# check clustering patterns
	merged_adata_sca.obs[cluster_key]=spots_cluster_labels
	merged_adata_sca.obs[cluster_key]=merged_adata_sca.obs[cluster_key].astype("category")
	print(merged_adata_sca.obs[cluster_key].value_counts())
	for tissue_section in tissue_section_list:
		section_adata=merged_adata_sca[merged_adata_sca.obs[merged_key]==tissue_section].copy()
		section_clusters=section_adata.obs[cluster_key].value_counts().index.tolist()
		section_cat_colors=[cat_color[i] for i in sorted(section_clusters)]
		fig_title=tissue_section+": heatmap clustering"
		fig_path=res_dir+"/"+tissue_section+"_"+cluster_key+"_nclusters="+str(best_k)+".png"
		os.makedirs(os.path.dirname(fig_path), exist_ok=True)
		cat_figure(input_adata=section_adata, x_col=x_col, y_col=y_col, fig_title=fig_title, fig_path=fig_path, color_key=cluster_key, cat_color=section_cat_colors, size=fig_scale/(section_adata.shape[0]**0.5), invert_x=invert_x, invert_y=invert_y)
	# generate the updated heatmap plots
	sorted_spot_indices=np.argsort(spots_clusters)
	sorted_col_indices=np.argsort(genes_clusters)
	sorted_gene_df=gene_df.iloc[sorted_spot_indices,:]
	sorted_columns=gene_df.columns[sorted_col_indices]
	sorted_gene_df=sorted_gene_df[sorted_columns]
	print(sorted_gene_df)
	fig_path=res_dir+"/"+cluster_key+"_upd.png"
	os.makedirs(os.path.dirname(fig_path), exist_ok=True)
	cluster_grid_upd=sns.clustermap(sorted_gene_df, cmap="coolwarm", row_cluster=False, col_cluster=False, yticklabels=False, xticklabels=True, figsize=(10,10))
	cluster_grid_upd.figure.savefig(fig_path)
	plt.clf()
	plt.close()
	# save subtype results
	subtype_clusters=merged_adata_sca.obs.copy()
	os.makedirs(res_dir+"/subtype_results", exist_ok=True)
	subtype_clusters.to_csv(res_dir+"/subtype_results/"+target_region+"_subtype_clusters.csv", index=True)
	# predicted labels within each tissue section
	for tissue_section in tissue_section_list:
		section_subtype_clusters=subtype_clusters.loc[subtype_clusters[merged_key]==tissue_section].copy()
		section_subtype_clusters.to_csv(res_dir+"/subtype_results/"+tissue_section+"_"+target_region+"_subtype_clusters.csv", index=True)
	return subtype_clusters


def shared_subtype_DE_genes(adata_dic,tissue_section_list,target_region,res_dir,overlap_cutoff,subtype_clusters,subtype_cluster_key="kmeans_clusters",pvals_adj=0.05,min_in_out_group_ratio=1,min_in_group_fraction=0.5,min_fold_change=1.1,merged_gene_num=15,individual_gene_num=35,x_col="pixel_x",y_col="pixel_y",fig_scale=2500,invert_x=False,invert_y=False,merged_key="sample", cnt_colormap="coolwarm"):
	adata_list=[]
	adata_sca_list=[]
	for tissue_section in tissue_section_list:
		test_gene=adata_dic[tissue_section]
		adata_list.append(test_gene)
		# normalization
		test_gene_sca=normalize_adata(test_gene)
		adata_sca_list.append(test_gene_sca)
	merged_adata=ad.concat(adata_list, axis=0, join="inner", label=merged_key, keys=tissue_section_list)
	merged_adata_sca=ad.concat(adata_sca_list, axis=0, join="inner", label=merged_key, keys=tissue_section_list)
	merged_adata.var["genes"]=merged_adata.var.index.tolist()
	merged_adata_sca.var["genes"]=merged_adata_sca.var.index.tolist()
	print(merged_adata.var)
	print(merged_adata_sca.var)
	print(merged_adata.obs[merged_key].value_counts())
	print(merged_adata_sca.obs[merged_key].value_counts())
	# add subtype cluster labels
	merged_adata.obs[subtype_cluster_key]=subtype_clusters.loc[merged_adata.obs.index,subtype_cluster_key].tolist()
	merged_adata_sca.obs[subtype_cluster_key]=subtype_clusters.loc[merged_adata_sca.obs.index,subtype_cluster_key].tolist()
	cluster_perct=pd.Series(subtype_clusters[subtype_cluster_key]).value_counts(normalize=True)
	#cluster_list=np.unique(subtype_clusters[subtype_cluster_key]).tolist()
	cluster_list=cluster_perct[cluster_perct>0.05].index.tolist()
	#----------------------- 1. Merged tissue sections to identify DE genes -----------------------
	d_g_merged={}
	print("----------------------------- Identify shared subtype DE genes by merging tissue sections -----------------------------")
	if len(cluster_list)>1:
		for cluster in cluster_list:
			merged_adata_sca.obs["target"]=(merged_adata_sca.obs[subtype_cluster_key]==cluster)*1
			df1=rank_genes_groups(input_adata=merged_adata_sca, target=1, label_col="target", non_target="rest", two_sides=False, logged=True)
			# filter genes based on filtering parameters
			#---------------------- df1 ----------------------
			df1_filtered=df1[(df1["pvals_adj"]<=pvals_adj) &
						 	 (df1["in_out_group_ratio"]>=min_in_out_group_ratio) &
						 	 (df1["in_group_fraction"]>=min_in_group_fraction) &
						     (df1["fold_change"]>=min_fold_change)]
			df1_filtered=df1_filtered.sort_values(by="fold_change", ascending=False)
			print(df1_filtered)	
			filtered_ngene=df1_filtered.shape[0]
			df1_genes=df1_filtered["genes"].tolist()[0:np.min([merged_gene_num, filtered_ngene])]
			print(df1_filtered.iloc[0:merged_gene_num])	
			d_g_merged[f"subtype{cluster}"]=df1_genes
		# check exp patterns
		for tissue_section in tissue_section_list:
			section_adata=merged_adata_sca[merged_adata_sca.obs[merged_key]==tissue_section].copy()
			for key, gene_list in d_g_merged.items():
				for g in gene_list:
					section_adata.obs[g]=section_adata[:,g].X.toarray().flatten()
					fig_title=tissue_section+": "+key+" ("+g+")"
					fig_path=res_dir+"/subtype_interpretations/merged_selection/"+tissue_section+"_"+target_region+"_subtype_DE_genes_merged_version_"+key+"_"+g+".png"
					os.makedirs(os.path.dirname(fig_path), exist_ok=True)
					con_figure(input_adata=section_adata, x_col=x_col, y_col=y_col, fig_title=fig_title, fig_path=fig_path, color_key=g, cnt_color=cnt_colormap, size=fig_scale/(section_adata.shape[0]**0.5), invert_x=invert_x, invert_y=invert_y)
	#----------------------- 2. Within each individual section and overlap to identify DE genes -----------------------
	d_g_ind={}
	d_g_section={}
	print("----------------------------- Identify shared subtype DE genes within each individual tissue sections -----------------------------")
	for tissue_section in tissue_section_list:
		d_g_cluster={}
		test_gene=merged_adata[merged_adata.obs[merged_key]==tissue_section].copy()
		section_cluster_perct=test_gene.obs[subtype_cluster_key].value_counts(normalize=True)
		#section_cluster_list=np.unique(test_gene.obs[subtype_cluster_key]).tolist()
		section_cluster_list=section_cluster_perct[section_cluster_perct>0.05].index.tolist()
		if len(section_cluster_list)>1:
			for cluster in section_cluster_list:
				test_gene.obs["target"]=(test_gene.obs[subtype_cluster_key]==cluster)*1
				df1=rank_genes_groups(input_adata=test_gene, target=1, label_col="target", non_target="rest", two_sides=False, logged=True)
				# filter genes based on filtering parameters
				#---------------------- df1 ----------------------
				df1_filtered=df1[(df1["pvals_adj"]<=pvals_adj) &
							 	 (df1["in_out_group_ratio"]>=min_in_out_group_ratio) &
							 	 (df1["in_group_fraction"]>=min_in_group_fraction) &
							     (df1["fold_change"]>=min_fold_change)]
				df1_filtered=df1_filtered.sort_values(by="fold_change", ascending=False)
				print(df1_filtered)	
				filtered_ngene=df1_filtered.shape[0]
				df1_genes=df1_filtered["genes"].tolist()[0:np.min([individual_gene_num, filtered_ngene])]
				print(df1_filtered.iloc[0:individual_gene_num])		
				d_g_cluster[f"subtype{cluster}"]=df1_genes
		d_g_section[tissue_section]=d_g_cluster
	# identify overlap genes
	for cluster in cluster_list:
		print(f"----------------------------- subtype{cluster} -----------------------------")
		cluster_union_genes_set=[]
		for tissue_section in tissue_section_list:
			if f"subtype{cluster}" in d_g_section[tissue_section]:
				cluster_union_genes_set=cluster_union_genes_set+d_g_section[tissue_section][f"subtype{cluster}"]
		cluster_gene_counts=pd.Series(cluster_union_genes_set).value_counts()
		print(cluster_gene_counts)
		final_cutoff=np.min([cluster_gene_counts.max(),overlap_cutoff])
		print(f"The final overlap counts cut-off is {final_cutoff}")
		overlap_genes=cluster_gene_counts[cluster_gene_counts>=final_cutoff].index.tolist()
		print(overlap_genes)
		d_g_ind[f"subtype{cluster}"]=overlap_genes
	# check gene exp patterns
	for tissue_section in tissue_section_list:
		section_adata=merged_adata_sca[merged_adata_sca.obs[merged_key]==tissue_section].copy()
		for key, gene_list in d_g_ind.items():
			for g in gene_list:
				section_adata.obs[g]=section_adata[:,g].X.toarray().flatten()
				fig_title=tissue_section+": "+key+" ("+g+")"
				fig_path=res_dir+"/subtype_interpretations/individual_selection/within_target_region/"+tissue_section+"_"+target_region+"_subtype_DE_genes_individual_version_"+key+"_"+g+".png"
				os.makedirs(os.path.dirname(fig_path), exist_ok=True)
				con_figure(input_adata=section_adata, x_col=x_col, y_col=y_col, fig_title=fig_title, fig_path=fig_path, color_key=g, cnt_color=cnt_colormap, size=fig_scale/(section_adata.shape[0]**0.5), invert_x=invert_x, invert_y=invert_y)
	return d_g_merged, d_g_ind



def normalize_adata(input_adata, method="min_max"):
	scaler_adata=MinMaxScaler()
	scaler_adata.fit(input_adata.X)
	X_scaled=scaler_adata.transform(input_adata.X)
	input_adata_sca=sc.AnnData(X_scaled)
	input_adata_sca.obs=input_adata.obs.copy()
	input_adata_sca.var=input_adata.var.copy()
	return input_adata_sca


def kmeans_clustering(features_matrix, n_clusters=5, random_state=0, kmeans_key="kmeans_clusters"):
	x=features_matrix.copy()
	kmeans=KMeans(n_clusters,random_state=random_state)
	y_pred=kmeans.fit_predict(x)
	print("========== KMeans Clustering Results ==========")
	print(pd.Series(y_pred).value_counts())
	return y_pred


def leiden_clustering(features_matrix, resolution, n_neighbors, random_state=0, leiden_key="leiden_clusters"):
	x=features_matrix.copy()
	tmp=sc.AnnData(x)
	sc.pp.neighbors(tmp,n_neighbors=n_neighbors,random_state=0)
	sc.tl.leiden(tmp,resolution=resolution,key_added=leiden_key)
	y_pred=tmp.obs[leiden_key].astype(int).to_numpy()
	print("========== Leiden Clustering Results ==========")
	print(pd.Series(y_pred).value_counts())
	n_clusters=len(np.unique(y_pred).tolist())
	# assert n_clusters>=2, "n_clusters < 2 -> Need to increase resolution!"
	return y_pred


def cat_figure(input_adata, x_col, y_col, fig_title, fig_path, color_key, cat_color, size, invert_x=False, invert_y=True):
	input_adata.obs[color_key]=input_adata.obs[color_key].astype("category")
	fig=sc.pl.scatter(input_adata, alpha=1, x=x_col, y=y_col, color=color_key, palette=cat_color, show=False, size=size)
	fig.set_aspect("equal","box")
	if invert_y==True:
		fig.invert_yaxis()
	if invert_x==True:
		fig.invert_xaxis()
	fig.set_title(fig_title)
	fig.figure.savefig(fig_path, dpi=100, bbox_inches="tight")
	del input_adata.uns[color_key+"_colors"]
	plt.clf()
	plt.close()


def con_figure(input_adata, x_col, y_col, fig_title, fig_path, color_key, cnt_color, size, invert_x=False, invert_y=True):
	fig=sc.pl.scatter(input_adata, alpha=1, x=x_col, y=y_col, color=color_key, color_map=cnt_color, show=False, size=size)
	fig.set_aspect("equal","box")
	if invert_y==True:
		fig.invert_yaxis()
	if invert_x==True:
		fig.invert_xaxis()
	fig.set_title(fig_title)
	fig.figure.savefig(fig_path,dpi=200, bbox_inches="tight")
	plt.clf()
	plt.close()



