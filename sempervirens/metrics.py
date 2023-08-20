# From HUNTRESS code

import numpy as np

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
                        # print("Seems NOT to be a PTree")
                        return False
                
    # print("Seems to be a PTree ...")
    return True

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
    return (n_adpairs, error_pairs, len(error_pairs), 1 - len(error_pairs)/n_adpairs)
    # return error_pairs                

def compareDL(M_orj,M_rec):  # Computes the Diff Lineage scores for M_rec given the ground truth matrix M_orj
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
    
    if d_pairs == 0:
        score = 1
    else:
        score = 1 - len(error_pairs)/d_pairs
    # print("Number of Diff pairs = ",d_pairs, "errors :",len(error_pairs), "score :", score)
    return (d_pairs, len(error_pairs), score)

def num_diffs(M1, M2):
    return np.sum(M1 != M2)