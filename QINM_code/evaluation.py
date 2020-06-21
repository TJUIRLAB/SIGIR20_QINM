from __future__ import division
import pandas as pd 
import subprocess
import platform,os
import sklearn
import numpy as np
import configure


FLAGS = configure.flags.FLAGS
FLAGS.flag_values_dict()

# get the number of relevance document
def get_query_qrels_number(qid):
	R = 0
	#web trec 2009-2011
	file = "../pre-process/"+FLAGS.data+"/qrels_1-150.txt"
	f = open(file)
	for line in f.readlines():
		lineList = line.split(" ")
		if int(lineList[0])<=50 and int(lineList[0]) == qid:
			if int(lineList[2])>0:
				R+=1
		elif int(lineList[0])>50 and int(lineList[3])>0 and int(lineList[0]) == qid:
			R+=1
	f.close()
	return R

def map_metric(group):
	group = sklearn.utils.shuffle(group,random_state =132)
	ap=0
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	correct_candidates=candidates[candidates["flag"]>=-15]
	qid = correct_candidates["query_ID"][0]
	R = get_query_qrels_number(qid)
	if R == 0:
		R = 1
	rank_list = correct_candidates["flag"]	
	is_one = 1
	index = 1
	if len(correct_candidates)==0:
		return 0
	for i in rank_list:
		if i > 0:
			ap += (1.0*is_one)/(1.0*index)
			is_one += 1
		index += 1
	return ap/(1.0*R)


def NDCG_metric(group):
	group = sklearn.utils.shuffle(group,random_state =132)
	group1 = sklearn.utils.shuffle(group,random_state =132)
	AtP=20
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	correct_candidates=candidates[candidates["flag"]>=-15]
	if len(correct_candidates) == 0:
		return 0
	rank_list = correct_candidates["flag"]
	p = 1
	DCG = 0
	index = 1
	for i in rank_list:
		if p >AtP:
			break
		DCG += (np.power(2.0,i)-1.0)/np.log2(index+1)
		index += 1
		p += 1

	candidates1 =group1.sort_values(by='flag',ascending=False)
	rank_list1 = candidates1["flag"]

	p = 1
	IDCG = 0
	index = 1
	for i in rank_list1:
		if p > AtP:
			break
		IDCG += (np.power(2.0,i)-1.0)/np.log2(index+1)
		index += 1
		p += 1

	if IDCG == 0:
		IDCG += 0.00001

	nDCG = float(DCG/IDCG)
	return nDCG

def NDCG_metric1(group):
	group = sklearn.utils.shuffle(group,random_state =132)
	group1 = sklearn.utils.shuffle(group,random_state =132)
	AtP=20
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	correct_candidates=candidates[candidates["flag"]>=-15]
	if len(correct_candidates)==0:
		return 0
	rank_list = correct_candidates["flag"]

	p = 1
	DCG = 0
	index = 1

	for i in rank_list:
		if p >AtP:
			break
		if index == 1:
			DCG += i
			index += 1
			p += 1
			continue		
		DCG += (1.0*i)/np.log2(index)
		index += 1
		p += 1

	candidates1 =group1.sort_values(by='flag',ascending=False)
	rank_list1 = candidates1["flag"]

	p = 1
	IDCG = 0
	index = 1
	for i in rank_list1:
		if p > AtP:
			break
		if index == 1:
			IDCG += i
			index += 1
			p += 1
			continue
		IDCG += (1.0*i)/np.log2(index)
		index += 1
		p += 1
	if IDCG == 0:
		IDCG += 0.00001
	nDCG = float(DCG/IDCG)
	return nDCG


def ERR_metric(group):
	AtP = 20

	if len(group) <AtP:
		AtP = len(group)

	group = sklearn.utils.shuffle(group,random_state =132)
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	correct_candidates=candidates[candidates["flag"]>=-15]
	if len(correct_candidates)==0:
		return 0
	rank_list = correct_candidates["flag"]
	gmax = rank_list.max()
	ERR = 0
	for r in range(1,AtP+1):
		pp_r = 1
		for i in range(1,r):
			R_i = float((np.power(2.0,rank_list[i-1])-1.0)/np.power(2.0,gmax))
			pp_r *= (1.0 - R_i)
		R_r = float((np.power(2.0,rank_list[r-1])-1.0)/np.power(2.0,gmax))
		pp_r *= R_r
		ERR += (1.0/r)*pp_r 

	return ERR

def p_metric(group):
	AtP = 20
	group = sklearn.utils.shuffle(group,random_state =132)
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	correct_candidates=candidates[candidates["flag"]>=-15]
	rank_list = correct_candidates["flag"]
	true_num = 0
	false_num = 0
	index = 1
	for i in rank_list:
		if index > AtP:
			break
		if i>0:
			true_num+=1
		else:
			false_num+=1
		index+=1
	p = float((1.0*true_num)/(1.0*AtP))
	return p

def evaluationBypandas(df,predicted):
	df["score"]=predicted
	map= df.groupby("query_content").apply(map_metric).mean()
	NDCG0 = df.groupby("query_content").apply(NDCG_metric).mean()
	NDCG1 = df.groupby("query_content").apply(NDCG_metric1).mean()
	ERR = df.groupby("query_content").apply(ERR_metric).mean()
	p = df.groupby("query_content").apply(p_metric).mean()
	return map,NDCG0,NDCG1,ERR,p


