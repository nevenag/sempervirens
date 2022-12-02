import numpy as np
import scipy.stats as st
import numpy as np
import scipy.stats as st
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from argparse import ArgumentParser
import itertools
from multiprocessing import Process, Queue
import multiprocessing as mp


#------------Testing and reading functions -------------------------------------------------------
def compareAD(M1,M2):      # Computes the AD scores for M2 given the ground truth matrix M1
    error_pairs=[]
    n_adpairs=0
    for i in range(M1.shape[1]):
#        print(i)
        for j in range(i,M1.shape[1]):
            cap1=M1[:,i]*M1[:,j]
            cap2=M2[:,i]*M2[:,j]
            if (np.sum(cap1)>0 and np.sum(M1[:,i]) != np.sum(M1[:,j])):
                n_adpairs=n_adpairs+1
                if (np.sum(cap2)==0):
                    error_pairs.append([i,j])
                else:
                    if (np.sum(M1[:,j])>np.sum(M1[:,i]) and np.sum(M2[:,j])<=np.sum(M2[:,i])):
                        error_pairs.append([i,j])
                    else:
                        if (np.sum(M1[:,i])>np.sum(M1[:,j]) and np.sum(M2[:,i])<=np.sum(M2[:,j])):
                            error_pairs.append([i,j])
                        #print(i,j,sum(M1[:,i]),sum(M1[:,j]),sum(M2[:,i]),sum(M2[:,j]))
    # print('Number of AD pairs = ',n_adpairs,"errors : ",len(error_pairs), "AD score = ", 1 - len(error_pairs)/n_adpairs)
    return error_pairs                

def compareDF(M_orj,M_rec):  # Computes the Diff Lineage scores for M_rec given the ground truth matrix M_orj
    error_pairs=[]
    d_pairs=0
    for i in range(M_orj.shape[1]):
        for j in range(i,M_orj.shape[1]):
            cap1=M_orj[:,i]*M_orj[:,j]
            cap2=M_rec[:,i]*M_rec[:,j]
            if np.sum(cap1)==0:
                d_pairs=d_pairs + 1
                if np.sum(cap2)>0:
                    error_pairs.append([i,j])
    # print("Number of Diff pairs = ",d_pairs, "errors :",len(error_pairs), "score :", 1-len(error_pairs)/d_pairs)
    return 

def ReadFile(filename):   # reads a matrix from file and returns it in the original type
    df = pd.read_csv(filename, sep='\t', index_col=0)
    M=df.values
    return M

def ReadFileNA(filename): # reads the file and fills the NA with 0's.
    df = pd.read_csv(filename, sep='\t', index_col=0)
    M=df.values
    
    NA_position=np.argwhere(M==3)
    for j in NA_position:
        M[j[0],j[1]]=0        

    return M.astype(bool)

def isPtree(matrix_in):   # brute force check if matrix_in is a pTree
    M=matrix_in.astype(bool)
    for j in range(M.shape[1]):
        for i in range(j,M.shape[1]):
            cap=M[:,i]*M[:,j]
            cap_size=np.sum(cap)
            Mi_size=np.sum(M[:,i])
            Mj_size=np.sum(M[:,j])
            if (cap_size != 0):
                if (cap_size != Mi_size):
                    if (cap_size != Mj_size):
                        return False
                
    # print("Seems to be a PTree ...")
    return True
#---------------------------------------------------------------------------------------------------------------

# Sempervirens Count based

# def com_prob(x1,x2,p01,p10): # both vectors are assumed to be probability vectors, each element shows the probability of that index being 1
#     px1=np.zeros(x1.shape)
#     px2=np.zeros(x2.shape)
#     j1=np.sum(x1)/x1.shape[0]     # estimated 1 ratio
#     p11i=j1*(1-p10)/(j1*(1-p10)+(1-j1)*p01)
#     p01i=(1-j1)*p01/(j1*(1-p10)+(1-j1)*p01)
#     #print(p11i,p01i)
#     px1=p11i*x1+p01i*(1-x1)
# #    print(px1)
#     j1=np.sum(x2)/x2.shape[0]     # estimated 1 ratio
#     p11i=j1*(1-p10)/(j1*(1-p10)+(1-j1)*p01)
#     p01i=(1-j1)*p01/(j1*(1-p10)+(1-j1)*p01)
#     #print(p11i,p01i)
#     px2=p11i*x2+p01i*(1-x2)
# #    print(px2)
#     #print(np.sum(px1),np.sum(px2))
#     x1supeqx2=np.prod(1-(1-px1)*px2)
#     x1subeqx2=np.prod(1-(1-px2)*px1)
#     x1eqx2=np.prod((1-px1)*(1-px2)+px1*px2)
#     x1disx2=np.prod(1-px1*px2)
#     #print(x1eqx2,x1disx2,x1subeqx2,x1supeqx2)
#     x1supx2=x1supeqx2-x1eqx2
#     x1subx2=x1subeqx2-x1eqx2
#     xsum=x1eqx2+x1disx2+x1subx2+x1supx2
#     x1eqx2=x1eqx2/xsum
#     x1disx2=x1disx2/xsum
#     x1subx2=x1subx2/xsum
#     x1supx2=x1supx2/xsum
# #    print(x1eqx2+x1disx2+x1subx2+x1supx2)
#     return(x1subx2,x1supx2,x1eqx2,x1disx2)


# def find_str(piv_index,Set_nodes,M_input,p01,p10): # tries to find the subtree that is connected to the pivot.
#     Search_set=Set_nodes.copy()
#     S=[]
#     S.append(piv_index)
#     p=piv_index
#     new_addition=1
#     while new_addition==1:
#         new_addition=0
        
#         for i in Search_set:
#             x1=M_input[:,p]
#             x2=M_input[:,i]
# #            [x1sx2,x2sx1,x1ex2,x1dx2]=com_prob(x1,x2,p01,p10)
# #             c=np.argmax([x1sx2+x1ex2,x2sx1+x1ex2,x1dx2])
# #            c=np.argmax([x1sx2,x2sx1+x1ex2,x1dx2])
#             c=np.argmax(com_prob(x1,x2,p01,p10))
#             if c == 0 and (i in S)==0:      # pivot is likely to be contained in the new set hence pivot is changed
#                 S.append(i)
#                 new_addition=1
#                 p=i
#                 break
#             elif c==1 and (i in S)==0:       # pivot is likely to contain the new set
#                 S.append(i)
#                 new_addition=1
#             elif c==2 and (i in S)==0:       # two sets are likely to be equal
#                 S.append(i)
#                 new_addition=1
#             else:                     # i is disjoint hence removed from the search set
#                 Search_set.remove(i)
                     
#     return [S,p]


# def find_stree(p_vec,Col_set,M_input,p01,p10): # tries to find the subtree that is connected to the pivot.
#     Search_set=Col_set.copy()
#     S=[]
    
#     #p=piv_index
#     new_addition=1
#     while new_addition==1:
#         new_addition=0
        
#         for i in Search_set:
#             x=M_input[:,i]
#             c=np.argmax(com_prob(p_vec,x,p01,p10))
            
#             if c == 0 and (i in S)==0:      # pivot is likely to be contained in the new set hence pivot is changed
#                 S.append(i)
#                 new_addition=1
#                 p=i
#                 break
#             elif c==1 and (i in S)==0:       # pivot is likely to contain the new set
#                 S.append(i)
#                 new_addition=1
#             elif c==2 and (i in S)==0:       # two sets are likely to be equal
#                 S.append(i)
#                 new_addition=1
#             else:                     # i is disjoint hence removed from the search set
#                 Search_set.remove(i)
#     return S    


def mlrefinementcol(M,M_noisy,p01,p10):  # this is for non missing element case M is reconstructed matrix, M_noisy is the noisy input
    M_output=M_noisy.copy()
    #print(M,M_noisy)
    for i in range(M_output.shape[1]):
        p=[]
        for j in range(M_output.shape[1]):
            #print(i,j,np.sum(M_output[:,i]>M[:,j])*np.log(p01)+np.sum(M_output[:,i]<M[:,j])*np.log(p10))
            p.append(np.sum(M_output[:,i]>M[:,j])*np.log(p01)+np.sum(M_output[:,i]<M[:,j])*np.log(p10))
    #print(M_output[:,i],M[:,np.argmax(p)], " are matched" )
        #print(i,np.argmax(p), " are matched" )
    
        M_output[:,i]=M[:,np.argmax(p)]
    
    return M_output

# Fin the number of ones in a column of size n that is most likely to give k ones one when measured.
def find_khat(k, n, fpr, fnr):
    from scipy.special import comb
    # khat: original number of ones in column
    # k: number of ones in measured column
    # n_fn: number of false negatives that occurr to go from khat to k
    # fnr: false negative rate
    # fpr: false positive rate

    # This is proportional to the probability that k 1s are measured in a column given khat 1 in the original column
    def pr_k_given_khat(khat, k, n, fpr, fnr):
        n_fn_upper = min(n - k, khat)
        n_fn_lower = max(0, khat - k)
        n_fn_s = range(n_fn_lower, n_fn_upper + 1)
        prs = [comb(khat, l)*comb(n - khat, l + k - khat) * np.power(fnr/(1-fpr), l) * np.power(1-fnr, khat - l) * np.power(fpr, l + k - khat) for l in n_fn_s]
        return sum(prs)
    
    # khats = range(n+1)
    # prs = np.array([pr_k_given_khat(khat, k, n, fpr, fnr) for khat in khats])
    # max_i = np.argmax(prs)
    # return max_i

    # Use that the probabilities increase and then decrease to realize we find the maximum early
    prev_pr = pr_k_given_khat(0, k, n, fpr, fnr)
    for khat in range(1, n + 1):
        pr = pr_k_given_khat(khat, k, n, fpr, fnr)
        if pr < prev_pr:
            return khat - 1
        prev_pr = pr
    return n
    

def rec_root(M_i,S,U,p01,p10): # reconstruct the pivot vector from the subtree estimated by S among U
    root_vec=np.zeros(M_i.shape[0])
#    print(S,U)
    root_in=np.zeros(M_i.shape[0])
    root_out=np.zeros(M_i.shape[0])
    kmax=0 # After first for loop, becomes max number of 1s that any column has

    for i in U:
        if i in S:
            kmax=max(kmax,np.sum(M_i[:,i]))
            root_in=root_in+M_i[:,i]
        else: 
            root_out=root_out+M_i[:,i]
    sort_index=np.flip(np.argsort(root_in)) # Go from row that had most 1s in subtree to least ones in subtree
    # sort_index=np.flip(np.argsort(root_in - root_out)) # Maximize margin

    # print(kmax*(1/((1-p10))), (kmax - p01*M_i.shape[1])/((1-p01-p10)), find_khat(kmax, M_i.shape[0], p01, p10))
    khat = find_khat(kmax, M_i.shape[0], p01, p10)
    ksum=0
    for j in sort_index:                     # flipping 0's that are seen the most
        if root_in[j] >= root_out[j] and root_in[j]>0:
            # margin = root_in[j] - root_out[j]
            # mf = margin/max(root_in[j], root_out[j])

            # thresh = khat
            thresh = 1.05*kmax/(1-p10)
            if ksum < thresh:
            # if ksum< kmax*(1/((1-p10)*mf)):       # 1.05        
                # This coefficient needs to be computed accurately. !!!!!!!! # try the estimation of priors + expectation
                root_vec[j]=1
                ksum=ksum+1
#    print(sum(root_vec),"\n",sum(root_in),"\n",sum(root_out))
    return root_vec

    # oct 13:
    # * maybe keep track of how many 0->1 and 1->0 overall, and how many 1s we think are in the overall original matrix, to do something similar to kmax
    # * redetermine what vecs to remove from original noisy matrix after each reconstruction
    #   * Masking issue?
    #   

def find_mlvec(p_vec,Col_set,M_in,M_mask,p01,p10): # returns most likely vector to the p_vec in Col_set
    mlv=Col_set[0]
    cp_current=0
#    print("finding the most likely vector !...")
    for i in Col_set:
        cp=np.sum(p_vec*(M_in*M_mask)[:,i])
        if cp>cp_current:
            mlv=i
            cp_current=cp
            
#    print(" Final : ", mlv,cp_current)        
    return mlv


def main_Rec(M_in, p01, p10, pnan): # main routine for reconstruction. The output can be fed to mlrefinement function for further processing

    M_masked=M_in.copy()

    # Account for missing entries by assuming they are 0s
    # Don't adjust FPR and FNR rates
    M_masked[M_masked == 3] = 0
    
    # # Account for missing entries by assuming they are 0s
    # # adjust FPR and FNR rates
    # # FPR stays as same
    # measured_num0 = np.sum(M_masked == 0)
    # measured_num1 = np.sum(M_masked == 1)
    # est_true_num1 = 1/(p01*p10 - (1-p01)*(1-p10)) * (p01*measured_num0 - (1-p01)*measured_num1)
    # est_true_num0 = 1/(1-p01) * (measured_num0 - p10*est_true_num1)
    # p10 = p10 + np.clip(pnan * est_true_num1 / (est_true_num1 + est_true_num0), 0, 1)
    # M_masked[M_masked == 3] = 0 

    M_rec_zero=np.zeros(M_in.shape)
    M_mask=np.ones(M_in.shape)                                         # a masking matrix
    S_rem=[[sum(M_in[:,i]),i] for i in range(M_in.shape[1])] # perhaps use expected number 1s
    S_rem.sort()
    S_rem.reverse()
    
    S_remsorted=[x[1] for x in S_rem]
#    print(S_remsorted)
    while S_remsorted != []:
        S_rem=[[np.sum(M_masked[:,i]),i] for i in S_remsorted]
        S_rem.sort()
        S_rem.reverse()
        S_remsorted=[x[1] for x in S_rem]
        i=S_remsorted[0]
        piv_rec=M_masked[:,i]

        # keep track of which subtrees each column is in. Due to masking, don't need to check whether columns we already know are in a separate subtree are in the same one again
        piv_rec_old = np.copy(piv_rec) 
        S = find_subset_countbased(M_masked,piv_rec,S_remsorted,p01,p10)
        piv_rec=rec_root(M_masked,S,S_remsorted,p01,p10)              # reconstruct the corresponding root
        while not np.all(piv_rec == piv_rec_old):
            piv_rec_old = np.copy(piv_rec)
            S = find_subset_countbased(M_masked,piv_rec,S_remsorted,p01,p10)
            piv_rec=rec_root(M_masked,S,S_remsorted,p01,p10)              # reconstruct the corresponding root

        #---Backsweep--------
        if S==[]:
            print(" Error S is empty ", p_indx, np.sum(piv_rec*M_masked[:,p_indx]))
            p_indx=i
            piv_rec=M_masked[:,i]
        Sfinal=[]
        # This is doing same as looking in subtree based on reconstructed pivot, but updating the masks as going
        for j in S_remsorted:
            if np.sum(M_masked[:,j]*piv_rec)>=np.sum(M_masked[:,j]*(1-piv_rec)): # More in common than in different
                M_masked[:,j]=M_masked[:,j]*piv_rec
                M_mask[:,j]=M_mask[:,j]*piv_rec
                Sfinal.append(j)
            else:
                M_masked[:,j]=M_masked[:,j]*(1-piv_rec)
                M_mask[:,j]=M_mask[:,j]*(1-piv_rec)
        if Sfinal==[]:
            print(" ERROR subtree is empty for pivot vector ") # Maybe not error? one case this happens is if piv_rec is all 0s
            print(f"piv rec all 0s? {np.all(piv_rec == 0)}")
            print(f"Num 1s in i: {np.sum(M_masked[:,i])}, {np.sum(M_in[:,i])}")
            #  print("S")
            # print(S)
            # print("S_remsorted")
            # print(S_remsorted)
            # print("i: ", i)
            # print("piv rec")
            # print(piv_rec)
            # print("M masked [:, i]")
            # print(M_masked[:,i])
            
            # print(" ERROR subtree is empty for pivot vector ", len(S),len(S_remsorted),i,piv_rec,M_masked[:,i])
            p_indx=i
            piv_rec=M_masked[:,i]
        else:
            p_indx=find_mlvec(piv_rec,Sfinal,M_masked,M_mask,p01,p10)
        M_rec_zero[:,p_indx]=piv_rec
        S_remsorted.remove(p_indx)
    
    return M_rec_zero

def find_subset_countbased(M_i,p_vec,Col_set,p01,p10):      # a simple count based subtree finder.
    S_return=[]
    p = p_vec
    for i in Col_set:
        if sum(p*M_i[:,i])>=sum((1-p)*M_i[:,i]):
            S_return.append(i)
    return S_return

