import numpy as np
import pandas as pd


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

def mlrefinementcol(M,M_noisy,p01,p10):  # this is for non missing element case M is reconstructed matrix, M_noisy is the noisy input
    M_output=M_noisy.copy()
    #print(M,M_noisy)
    for i in range(M_output.shape[1]):
        p=[]
        for j in range(M_output.shape[1]):
            #print(i,j,np.sum(M_output[:,i]>M[:,j])*np.log(p01)+np.sum(M_output[:,i]<M[:,j])*np.log(p10))
            p.append(np.sum(M_output[:,i] > M[:,j]) * np.log(p01) + np.sum(M_output[:,i] < M[:,j]) * np.log(p10))
    #print(M_output[:,i],M[:,np.argmax(p)], " are matched" )
        #print(i,np.argmax(p), " are matched" )
    
        M_output[:,i]=M[:,np.argmax(p)]
    
    return M_output

#-------- Row related functions ------------------------------------------------------

def reconstruct_row(mat, subtree, sorted_row_set, fpr, fnr):
    # TODO: parent node HAS to show in every child
    root = np.zeros(mat.shape[1])
    
    kmin = np.min(np.sum(mat[subtree, :], axis = 1)) # Maximum number of ones in any row.
    subtree_in = np.sum(mat[subtree, :], axis = 0)
    subtree_out = np.sum(mat[sorted_row_set, :], axis = 0) - subtree_in
    assert np.all(subtree_in.shape == root.shape)
    assert np.all(subtree_out.shape == root.shape)

    thresh = kmin # 1.05 * kmin / (1 - fnr)
    print(kmin)
    # print('thresh: ', thresh)
    # print('subtree in: ', subtree_in)
    sort_index = np.flip(np.argsort(subtree_in)) # Go from col with most to least number of 1s in subtree.
    ksum = 0 # Number of 1s placed in root so far.
    for j in sort_index:
        if ksum < thresh and subtree_in[j] > subtree.size - subtree_in[j]:
            root[j] = 1
            ksum += 1
    
    assert np.all(np.logical_or(root == 0, root == 1))
    return root.astype(int)

def get_root(mat, piv_col, fpr, fnr):
    return reconstruct_row(mat, piv_col, np.ones_like(piv_col), fpr, fnr)

    row_indices = piv_col.reshape((piv_col.size, 1))
    histogram = np.sum(row_indices * mat, axis = 0)
    subtree_size_measured = np.sum(row_indices) # The number of rows that should have 1s if the col is present in the root.
    # orig_subtree_size_est = (subtree_size_measured - fpr*mat.shape[0]) / (1 - fpr - fnr)
    # threshold = (1 - fnr)**2 * orig_subtree_size_est + fpr**2 * (mat.shape[0] - orig_subtree_size_est)
    threshold = subtree_size_measured * (1 - fnr)
    # threshold = orig_subtree_size_est * (1 - fnr)
    root = (histogram >= threshold).astype(int)
    # print('root', np.sum(root), root)
    return root

# Is row A a descendant of row B?
def is_descendant(a, b, num_cols, fpr, fnr):
    # return np.sum(a * b) / (1 - fnr) >= np.sum(b)
    # return np.all(a*b == b) # Would result in no changes to descendants
    est_size_b = (np.sum(b) - fpr*num_cols) / (1 - fpr - fnr)
    return np.sum(a * b) / (1 - fnr) >= est_size_b
    # return np.sum(a*b) >= (1-fnr)**2 * est_size_b + fpr**2 * (num_cols - est_size_b)

def find_rootnode(M_i,Col_set,p10):
    x_repeat=sum(M_i[:,Col_set])
#    x_repeat=sum(M_i)
    x_max=max(x_repeat)
    x_recn=np.zeros(M_i.shape[1])
 
    x_recn[Col_set]=x_repeat>=x_max*(1-p10) + 0
#    x_recn=x_repeat>=x_max*(1-p10) + 0
    for i in range(M_i.shape[0]):
        
#        if x_recn.dot(M_i[i,:])>=sum(x_recn)/2 and x_recn.dot(M_i[i,:])<sum(x_recn):
        if x_recn.dot(M_i[i,:])>=sum(x_recn)*(0.5):
            #print(x_recn.dot(M_i[i,:]),sum(x_recn))
            M_i[i,:]=(M_i[i,:]+x_recn)>0 + 0
            # if x_recn.dot(M_grnd[i,:])<sum(x_recn):
            #     print(x_recn.dot(M_grnd[i,:]),sum(x_recn))
#         else:
#             if sum(M_i[i,:]*(1-x_recn) + 0)<sum(M_i[i,:]):
#                 print(sum(M_i[i,:]*(1-x_recn) + 0),sum(M_i[i,:]),sum(M_grnd[i,:]))


#-------------------------------------------------------------------------------------

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
    

def rec_root(M_i,S,U,p01,p10): # reconstruct t>he pivot vector from the subset estimated by S among U
    root_vec = np.zeros(M_i.shape[0])
#    print(S,U)
    root_in = np.zeros(M_i.shape[0])
    root_out = np.zeros(M_i.shape[0])
    kmax = 0 # After first for loop, becomes max number of 1s / number of non nans (3s) that any column has

    # 2022-12-01: right now this type of root = root + m * (m != 3) effectively is considering missing elements to be 0s in terms of summing them up
    #    perhaps they should be given some minor amount of positive value based on an estimate of prior elem being 1?
    for i in U:
        if i in S:
            kmax = max(kmax, np.sum(M_i[:, i] == 1))
            root_in = root_in + M_i[:,i] * (M_i[:, i] != 3)
        else: 
            root_out = root_out + M_i[:,i] * (M_i[:, i] != 3)
    sort_index = np.flip(np.argsort(root_in)) # Go from row that had most 1s in subset to least ones in subset
    # sort_index=np.flip(np.argsort(root_in - root_out)) # Maximize margin

    # print(kmax*(1/((1-p10))), (kmax - p01*M_i.shape[1])/((1-p01-p10)), find_khat(kmax, M_i.shape[0], p01, p10))
    # khat = find_khat(kmax, M_i.shape[0], p01, p10)
    ksum=0
    for j in sort_index:                     # flipping 0's that are seen the most
        if root_in[j] >= root_out[j] and root_in[j]>0:
            # margin = root_in[j] - root_out[j]
            # mf = margin/max(root_in[j], root_out[j])

            # thresh = khat
            # thresh = 1.05*kmax/(1-p10)
            if True: # ksum < thresh:
            # if ksum< kmax*(1/((1-p10)*mf)):       # 1.05        
                # This coefficient needs to be computed accurately. !!!!!!!! # try the estimation of priors + expectation
                root_vec[j] = 1
                ksum = ksum + 1
#    print(sum(root_vec),"\n",sum(root_in),"\n",sum(root_out))
    assert np.all(np.logical_or(root_vec == 0, root_vec == 1))
    return root_vec

    # oct 13:
    # * maybe keep track of how many 0->1 and 1->0 overall, and how many 1s we think are in the overall original matrix, to do something similar to kmax
    # * redetermine what vecs to remove from original noisy matrix after each reconstruction
    #   * Masking issue?
    #   

def find_mlvec(p_vec,Col_set,M_in,M_mask,p01,p10): # returns most likely vector to the p_vec in Col_set
    mlv=Col_set[0]
    cp_current=0
    for i in Col_set:
        cp=np.sum(p_vec*(M_in*M_mask)[:,i])
        if cp>cp_current:
            mlv=i
            cp_current=cp    
    return mlv


def main_Rec(M_in, p01, p10, pnan): # main routine for reconstruction. The output can be fed to mlrefinement function for further processing
    assert np.all(np.logical_or(np.logical_or(M_in == 0, M_in == 1), M_in == 3))

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
    M_mask=np.ones(M_in.shape) # a masking matrix
    S_rem=[[np.sum(M_in[:, i] == 1) / np.sum(M_in[:, i] != 3), i] for i in range(M_in.shape[1])] # perhaps use expected number 1s
    S_rem.sort()
    S_rem.reverse()
    
    S_remsorted=[x[1] for x in S_rem]
#    print(S_remsorted)
    while S_remsorted != []:
        S_rem = [[np.sum(M_in[:, i] == 1) / np.sum(M_in[:, i] != 3), i] for i in S_remsorted]
        S_rem.sort()
        S_rem.reverse()
        S_remsorted = [x[1] for x in S_rem]
        i = S_remsorted[0]
        piv_rec = M_masked[:, i]

        # TODO: keep track of which subsets each column is in. Due to masking, don't need to check whether columns we already know are in a separate subset are in the same one again
        piv_rec_old = np.copy(piv_rec) 
        S = find_subset_countbased(M_masked, piv_rec, S_remsorted, p01, p10)
        if len(S) != 0:
            find_rootnode(M_masked, S, p10)
        piv_rec = rec_root(M_masked, S, S_remsorted, p01, p10)              # reconstruct the corresponding root
        while not np.all(piv_rec == piv_rec_old):
            piv_rec_old = np.copy(piv_rec)
            S = find_subset_countbased(M_masked, piv_rec, S_remsorted, p01, p10)
            if len(S) != 0:
                find_rootnode(M_masked, S, p10)
            piv_rec = rec_root(M_masked, S, S_remsorted, p01, p10)              # reconstruct the corresponding root


        # # Get root node (row)
        # root_node = get_root(M_masked, piv_rec, p01, p10)
        # # Enforce root on each row of matrix
        # for row_i in range(M_masked.shape[0]):
        #     # If row_i is a descendant of root_node, then add 1s to row_i so it is a superset of root_node
        #     if is_descendant(M_masked[row_i, :], root_node, M_masked.shape[1], p01, p10):
        #         M_masked[row_i, :] = np.logical_or(M_masked[row_i, :], root_node).astype(int)

        #---Backsweep--------
        if S==[]:
            # print("HAVENT HANDLED THIS FOR WHEN M_IN CONTAINS 3s (nans)")
            print(" Error S is empty ", p_indx, np.sum(piv_rec*(piv_rec != 3)*M_masked[:, p_indx]*(M_masked[:, i] != 3)))
            p_indx = i  
            piv_rec = M_masked[:, i] * (M_masked[:, i] != 3)
        Sfinal=[]
        # This is doing same as looking in subsets based on reconstructed pivot, but updating the masks as going
        assert np.all(np.logical_or(piv_rec == 0, piv_rec == 1))
        for j in S_remsorted:
            if np.sum(piv_rec*M_masked[:, j]*(M_masked[:, j] != 3)) >= np.sum((1-piv_rec)*M_masked[:, j]*(M_masked[:, j] != 3)): # More in common than in different
                M_masked[:, j] = M_masked[:, j] * piv_rec
                M_mask[:, j] = M_mask[:, j] * piv_rec
                Sfinal.append(j)
            else:
                M_masked[:, j] = M_masked[:, j] * (1-piv_rec)
                M_mask[:, j] = M_mask[:, j] * (1-piv_rec)
        if Sfinal==[]:
            print(" ERROR subset is empty for pivot vector ") # Maybe not error? one case this happens is if piv_rec is all 0s
            print(f"piv rec all 0s? {np.all(piv_rec == 0)}")
            print(f"Num 1s in i: {np.sum(M_masked[:, i] == 1)}, {np.sum(M_in[:, i] == 1)}")
            
            # print(" ERROR subset is empty for pivot vector ", len(S),len(S_remsorted),i,piv_rec,M_masked[:,i])
            p_indx = i
            piv_rec = M_masked[:, i] * (M_masked[:, i] != 3)
        else:
            p_indx = find_mlvec(piv_rec, Sfinal, M_masked, M_mask, p01, p10)
        M_rec_zero[:, p_indx] = piv_rec
        S_remsorted.remove(p_indx)
    
    assert np.all(np.logical_or(M_rec_zero == 0, M_rec_zero == 1))

    return M_rec_zero

def find_subset_countbased(M_i, p_vec, Col_set, p01, p10): # A simple count based subset finder.
    # TODO: update pivot vector after each column added to subset
    S_return=[]
    p = p_vec
    for i in Col_set:
        if np.sum(p*(p != 3)*M_i[:, i]*(M_i[:, i] != 3)) >= np.sum((1-p)*((1-p) != 3)*M_i[:, i]*(M_i[:, i] != 3)):
            S_return.append(i)
    return S_return

