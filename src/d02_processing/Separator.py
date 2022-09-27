import numpy as np
import scipy.stats as st
# Sempervirens 
def computeprob(x1,x2): # both vectors are assumed to be probability vectors, each element shows the probability of that index being 1
    x1supeqx2=np.prod(1.0-(1.0-x1)*x2)
    x1subeqx2=np.prod(1.-(1.-x2)*x1)
    x1eqx2=np.prod((1.-x1)*(1.-x2)+x1*x2)
    #x1disx2=1+x1eqx2-x1subeqx2-x1supeqx2
    x1disx2=np.prod(1.-x1*x2)
    x1supx2=x1supeqx2-x1eqx2
    x1subx2=x1subeqx2-x1eqx2
    xsum=x1eqx2+x1disx2+x1subx2+x1supx2
    x1eqx2=x1eqx2/xsum
    x1disx2=x1disx2/xsum
    x1subx2=x1subx2/xsum
    x1supx2=x1supx2/xsum
    #p_x2containsx1=x1subx2*(x1subx2>0)/
    
    return [x1subeqx2*(x1subeqx2>0),x1supeqx2*(x1supeqx2>0),x1eqx2*(x1eqx2>0),x1disx2*(x1disx2>0)]
#    return [x1subx2*(x1subx2>0),x1supx2*(x1supx2>0),x1eqx2*(x1eqx2>0),x1disx2*(x1disx2>0)]

def findsubtree(piv_index,Set_nodes,Matrix_input): # tries to find the subtree that is connected to the pivot.
    S=[]
    S.append(piv_index)
    p=piv_index
    new_addition=1
    while new_addition==1:
        new_addition=0
        
        for i in Set_nodes:
            x1=Matrix_input[:,p]
            x2=Matrix_input[:,i]
            [x1sx2,x2sx1,x1ex2,x1dx2]=computeprob(x1,x2)
            c=np.argmax([x1sx2+x1ex2,x2sx1+x1ex2,x1dx2])
            if c == 0 and (i in S)==0:
                S.append(i)
                new_addition=1
                #print("New addition ",i, " because it contains ",p," pivot UPDATED!")
                p=i
                break
            elif c==1 and (i in S)==0:
                S.append(i)
                new_addition=1
                #print("New addition ",i, " because it is a subset of ",p)
            #else:
            #    print(p," and ",i," are disjoint!")
                
                
            #print(S)
    return S
def Prob_matrix_converter(M_i,p01,p10): # p10 prob that original matrix had a 1 but noisy had a 0, p01 similar, M_input is the noisy matrix
    M_prob=np.zeros(M_i.shape)
    for j in range(M_i.shape[1]):
    #j1=(np.sum(M_input[:,j])-M_noise.shape[0])/(1-p10-p01)
    #j1=j1/M_input.shape[0]
        #j1=np.sum(M_input[:,j])/M_input.shape[0]
        j1=(np.sum(M_i)-p01*(M_i.shape[0]*M_i.shape[1]))/(1-p01-p10)
        j1=j1/(M_i.shape[0]*M_i.shape[1])
        
        # print(j1)
        pin1out1=j1*(1-p10)
        pin1out0=j1*p10
        pin0out1=(1-j1)*p01
        pin0out0=(1-j1)*(1-p01)
        for i in range(M_i.shape[0]):
            if M_i[i,j]==1:
                M_prob[i,j]=pin1out1/(pin1out1+pin0out1)
            if M_i[i,j]==0:
                M_prob[i,j]=pin1out0/(pin1out0+pin0out0)
#     for i in range(M_i.shape[0]):
#     #j1=(np.sum(M_input[:,j])-M_noise.shape[0])/(1-p10-p01)
#     #j1=j1/M_input.shape[0]
#         #j1=np.sum(M_input[:,j])/M_input.shape[0]
#         j1=(np.sum(M_i[i,:])-p01*(M_i.shape[1]))/(1-p01-p10)
#         j1=j1/(M_i.shape[1])
        
#         print(j1)
#         pin1out1=j1*(1-p10)
#         pin1out0=j1*p10
#         pin0out1=(1-j1)*p01
#         pin0out0=(1-j1)*(1-p01)
#         for j in range(M_i.shape[1]):
#             if M_i[i,j]==1:
#                 M_prob[i,j]=pin1out1/(pin1out1+pin0out1)
#             if M_i[i,j]==0:
#                 M_prob[i,j]=pin1out0/(pin1out0+pin0out0)
    return M_prob

def probcomputeraw(r,S,p01,p10):
    r_prob=np.zeros(r.shape[0])
    pi1=0
    for j in S:
        pi1=pi1+r[j]
    pi1=pi1/len(S)
    pin1out1=pi1*(1-p10)
    pin1out0=pi1*p10
    pin0out1=(1-pi1)*p01
    pin0out0=(1-pi1)*(1-p01)
    for i in S:
        if r[i]==1:
            r_prob[i]=pin1out1/(pin1out1+pin0out1)
        else:
            r_prob[i]=pin1out0/(pin1out0+pin0out0)
    return r_prob

        
    
    
# def Prob_matrix_converter(M_input,p01,p10): # p10 prob that original matrix had a 1 but noisy had a 0, p01 similar, M_input is the noisy matrix
#     for j in range(M_input.shape[1]):
#     #j1=(np.sum(M_input[:,j])-M_noise.shape[0])/(1-p10-p01)
#     #j1=j1/M_input.shape[0]
        
#         j1=np.sum(M_input[:,j])/M_input.shape[0]
#         pin1out1=j1*(1-p10)
#         pin1out0=j1*p10
#         pin0out1=(1-j1)*p01
#         pin0out0=(1-j1)*(1-p01)
#         for i in range(M_input.shape[0]):
#             if M_input[i,j]==1:
#                 M_prob[i,j]=pin1out1/(pin1out1+pin0out1)
#             if M_input[i,j]==0:
#                 M_prob[i,j]=pin1out0/(pin1out0+pin0out0)
#     return M_prob

# def reconstruct_root(M_prob,S,U): # reconstruct the pivot vector from the subtree estimated by S among U
#     root_vec=np.zeros(M_prob.shape[0])
#     if U != S:
#         for i in range(M_prob.shape[0]):
#             probi=1  # probability that ith element is 0
#             for j in S:  
#                 probi=probi*(1-M_prob[i,j])
#             probi2=1 # probability that ith element is 1
#             for j in U:
#                 if (j in S)==0:
#                     probi2=probi2*(1-M_prob[i,j])
#         #print(probi,probi2)

#             p1=(1-probi)*probi2
#             p2=probi*(1-probi2)
#             p3=probi*probi2
#             print(p1,p2,p3)
#             case=np.argmax([p1,p2,p3])
#             if case==1 or case == 2:
#                 root_vec[i]=0
#             else:
#                 root_vec[i]=1
#     else:
#         for i in range(M_prob.shape[0]):
#             probi=1  # probability that ith element is 0
#             for j in S:  
#                 probi=probi*(1-M_prob[i,j])
#             if probi>0.5:
#                 root_vec[i]=0
#             else:
#                 root_vec[i]=1
        
        
#     return root_vec

def reconstruct_root(M_i,S,U, p01, p10): # reconstruct the pivot vector from the subtree estimated by S among U
    root_vec=np.zeros(M_i.shape[0])
    if U != S:
        for i in range(M_i.shape[0]):
            pi=probcomputeraw(M_i[i,:],U,p01,p10)
            probi=1  # probability that ith element is 0
            for j in S:  
                probi=probi*(1-pi[j])
            probi2=1 # probability that ith element is 1
            for j in U:
                if (j in S)==0:
                    probi2=probi2*(1-pi[j])
        #print(probi,probi2)

            p1=(1-probi)*probi2
            p2=probi*(1-probi2)
            p3=probi*probi2
            # print(p1,p2,p3)
            case=np.argmax([p1,p2,p3])
            if case==1 or case == 2:
                root_vec[i]=0
            else:
                root_vec[i]=1
    else:
        for i in range(M_i.shape[0]):
            pi=probcomputeraw(M_i[i,:],U,p01,p10)
            probi=1  # probability that ith element is 0
            for j in S:  
                probi=probi*(1-pi[j])
            if probi>0.5:
                root_vec[i]=0
            else:
                root_vec[i]=1
        
        
    return root_vec
# def reconstruct_root(M_prob,S,U): # reconstruct the pivot vector from the subtree estimated by S among U
#     root_vec=np.zeros(M_prob.shape[0])
#     for i in range(M_prob.shape[0]):
#         probi=1  # probability that ith element is 0
#         for j in S:  
#             probi=probi*(1-M_prob[i,j])
#         probi2=1 # probability that ith element is 1
#         for j in U:
#             if (j in S)==0:
#                 probi2=probi2*(1-M_prob[i,j])
#         print(probi,probi2)
#         if probi>probi2:
#             root_vec[i]=0
#         else:
#             root_vec[i]=1
#     print(root_vec)
    
    
#def Convert_Probability_Matrix(M_input,p10,p01):
#    for i in range(M_input.shape[1]):
        
    
    
    
# def findsubtree(piv_index,Set_nodes,Matrix_input): # tries to find the subtree that is connected to the pivot.
#     S=[]
#     S.append(piv_index)
    
#     new_addition=1
#     while new_addition==1:
#         new_addition=0
        
#         for i in Set_nodes:
#             for p in S:
#                 x1=Matrix_input[:,p]
#                 x2=Matrix_input[:,i]
#                 [x1sx2,x2sx1,x1ex2,x1dx2]=computeprob(x1,x2)
#                 c=np.argmax([x1sx2+x1ex2,x2sx1+x1ex2,x1dx2])
#                 if (c == 0 or c==1) and (i in S)==0:
#                     S.append(i)
#                     new_addition=1
#                     print("New addition ",i, " because it is related to ",p)
#                     break
#             print(S)
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
        # print(i,np.argmax(p), " are matched" )
    
        M_output[:,i]=M[:,np.argmax(p)]
    
    return M_output

def rec_root(M_i,S,U,p01,p10):  # A counting based reconstructor. Reconstruct the pivot vector from the subtree estimated by S among U
    root_vec=np.zeros(M_i.shape[0])
#    print(S,U)
    root_in=np.zeros(M_i.shape[0])
    root_out=np.zeros(M_i.shape[0])
    kmax=0
    for i in U:
        
        if i in S:
            kmax=max(kmax,np.sum(M_i[:,i]))
            root_in=root_in+M_i[:,i]
        else: 
            root_out=root_out+M_i[:,i]
    sort_index=np.flip(np.argsort(root_in))
    ksum=0
    for j in sort_index:                     # flipping 0's that are seen the most
        if root_in[j]>root_out[j]:
            if ksum<kmax/(1-p10):
                root_vec[j]=1
                ksum=ksum+1
#    print(sum(root_vec),"\n",sum(root_in),"\n",sum(root_out))
       
        
    return root_vec

def mainReconstructor(M_in,p01,p10): # main rutine for reconstruction. The output can be fed to mlrefinement function for further processing
    M_rec=M_in.copy()
    M_p=Prob_matrix_converter(M_in,p01,p10)        # estimate the probability of the 
#    print(M_p)
    n_col = M_in.shape[1]
    S_rem=[[sum(M_p[:,i]),i] for i in range(n_col)]
    S_rem.sort()
    S_rem.reverse()
    S_remsorted=[x[1] for x in S_rem]

    while S_remsorted != []:
        i=S_remsorted[0]
        S=findsubtree(i,S_remsorted,M_in)           # find the subtree
        # piv_rec=reconstruct_root(M_p,S,S_remsorted, p01, p10) # reconstruct the corresponding root
        piv_rec = rec_root(M_p, S, S_remsorted, p01, p10)
        # print(M_rec[:,i],piv_rec,S_remsorted,S)
        M_rec[:,i]=piv_rec
        S_remsorted.remove(i)
    
    return M_rec
