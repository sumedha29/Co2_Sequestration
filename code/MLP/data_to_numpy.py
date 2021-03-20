from sklearn.utils import shuffle
import os
import numpy as np
import glob
import collections

def numpy_single(all_pressures,all_saturations,all_permeabilities,all_porosities,all_surf_inj_rate_series,all_surf_prod_rate_series,Ks,Rs,tr_var):
    features1 = []
    target1 = []

    for i in range(all_pressures.shape[0]):
        for j in range(all_pressures.shape[1]):
            for k in range(all_pressures.shape[2]):
                for l in range(all_pressures.shape[3]):
                    for m in range(all_pressures.shape[4]):
                        features1.append([(k/24),(l/24),(m/2),(j/71),(all_permeabilities[i][j][k][l][m]),(all_porosities[i][j][k][l][m]),(all_surf_inj_rate_series[i][j])])
                        try:
                            target1.append(tr_var[i][j][k][l][m])
                        except:
                            target1.append(tr_var[i][j])

    features1 = np.array(features1)
    target1 = np.array(target1)
    target1 = np.expand_dims(target1, axis=1)
    print(features1.shape)
    print(target1.shape)


    # READING


    features1_tr1 = features1[:2*72*25*25*3]
    features1_te1 = features1[2*72*25*25*3:4*72*25*25*3]
    features1_tr2 = features1[4*72*25*25*3:11*72*25*25*3]
    features1_te2 = features1[11*72*25*25*3:12*72*25*25*3]
    features1_tr3 = features1[12*72*25*25*3:]

    features1_tr = np.concatenate((features1_tr1,features1_tr2,features1_tr3))
    features1_te = np.concatenate((features1_te1,features1_te2))

    target1_tr1 = target1[:2*72*25*25*3]
    target1_te1 = target1[2*72*25*25*3:4*72*25*25*3]
    target1_tr2 = target1[4*72*25*25*3:11*72*25*25*3]
    target1_te2 = target1[11*72*25*25*3:12*72*25*25*3]
    target1_tr3 = target1[12*72*25*25*3:]

    target1_tr = np.concatenate((target1_tr1,target1_tr2,target1_tr3))
    target1_te = np.concatenate((target1_te1,target1_te2))

    
    features1_tr,target1_tr = shuffle(features1_tr,target1_tr, random_state=0)

    print(features1_tr.shape)
    print(target1_tr.shape)
    print(features1_te.shape)
    print(target1_te.shape)
    
    return features1_tr,target1_tr,features1_te,target1_te


def numpy_multi(all_pressures,all_saturations,all_permeabilities,all_porosities,all_surf_inj_rate_series,all_surf_prod_rate_series,Ks,Rs):
    features1 = []
    features2 = []
    target1 = []
    target2 = []
    target3 = []
    print('start')
    for i in range(all_pressures.shape[0]):
        for j in range(all_pressures.shape[1]):
            for k in range(all_pressures.shape[2]):
                for l in range(all_pressures.shape[3]):
                    for m in range(all_pressures.shape[4]):
                        features1.append([(k/24),(l/24),(m/2),(j/71),(all_permeabilities[i][j][k][l][m]),(all_porosities[i][j][k][l][m]),(all_surf_inj_rate_series[i][j])])
                        features2.append([(j/71),(Ks[i]/2),(Rs[i]/7)])
                        target1.append(all_saturations[i][j][k][l][m])
                        target2.append(all_pressures[i][j][k][l][m])
                        target3.append(all_surf_prod_rate_series[i][j])
                        

    features1 = np.array(features1)
    features2 = np.array(features2,dtype='float32')
    target1 = np.array(target1)
    target1 = np.expand_dims(target1, axis=1)
    target2 = np.array(target2)
    target2 = np.expand_dims(target2, axis=1)
    target3 = np.array(target3)
    target3 = np.expand_dims(target3, axis=1)
    print('end')
    print(features1.shape)
    print(features2.shape)
    print(target1.shape)
    print(target2.shape)
    print(target3.shape)

    features1_tr1 = features1[:2*72*25*25*3]
    features1_te1 = features1[2*72*25*25*3:4*72*25*25*3]
    features1_tr2 = features1[4*72*25*25*3:11*72*25*25*3]
    features1_te2 = features1[11*72*25*25*3:12*72*25*25*3]
    features1_tr3 = features1[12*72*25*25*3:]

    features1_tr = np.concatenate((features1_tr1,features1_tr2,features1_tr3))
    features1_te = np.concatenate((features1_te1,features1_te2))

    features2_tr1 = features2[:2*72*25*25*3]
    features2_te1 = features2[2*72*25*25*3:4*72*25*25*3]
    features2_tr2 = features2[4*72*25*25*3:11*72*25*25*3]
    features2_te2 = features2[11*72*25*25*3:12*72*25*25*3]
    features2_tr3 = features2[12*72*25*25*3:]

    features2_tr = np.concatenate((features2_tr1,features2_tr2,features2_tr3))
    features2_te = np.concatenate((features2_te1,features2_te2))

    target1_tr1 = target1[:2*72*25*25*3]
    target1_te1 = target1[2*72*25*25*3:4*72*25*25*3]
    target1_tr2 = target1[4*72*25*25*3:11*72*25*25*3]
    target1_te2 = target1[11*72*25*25*3:12*72*25*25*3]
    target1_tr3 = target1[12*72*25*25*3:]

    target1_tr = np.concatenate((target1_tr1,target1_tr2,target1_tr3))
    target1_te = np.concatenate((target1_te1,target1_te2))

    target2_tr1 = target2[:2*72*25*25*3]
    target2_te1 = target2[2*72*25*25*3:4*72*25*25*3]
    target2_tr2 = target2[4*72*25*25*3:11*72*25*25*3]
    target2_te2 = target2[11*72*25*25*3:12*72*25*25*3]
    target2_tr3 = target2[12*72*25*25*3:]

    target2_tr = np.concatenate((target2_tr1,target2_tr2,target2_tr3))
    target2_te = np.concatenate((target2_te1,target2_te2))

    target3_tr1 = target3[:2*72*25*25*3]
    target3_te1 = target3[2*72*25*25*3:4*72*25*25*3]
    target3_tr2 = target3[4*72*25*25*3:11*72*25*25*3]
    target3_te2 = target3[11*72*25*25*3:12*72*25*25*3]
    target3_tr3 = target3[12*72*25*25*3:]

    target3_tr = np.concatenate((target3_tr1,target3_tr2,target3_tr3))
    target3_te = np.concatenate((target3_te1,target3_te2))
    features1_tr,features2_tr,target1_tr,target2_tr,target3_tr = shuffle(features1_tr,features2_tr,target1_tr,target2_tr,target3_tr, random_state=0)

    print(features1_tr.shape)
    print(features2_tr.shape)
    print(target1_tr.shape)
    print(target2_tr.shape)
    print(target3_tr.shape)
    print(features1_te.shape)
    print(features2_te.shape)
    print(target1_te.shape)
    print(target2_te.shape)
    print(target3_te.shape)
    
    return features1_tr,features2_tr,target1_tr,target2_tr,target3_tr,features1_te,features2_te,target1_te,target2_te,target3_te



def numpy_read_physics(all_pressures,all_saturations,all_permeabilities,all_porosities,all_surf_inj_rate_series,all_surf_prod_rate_series,Ks,Rs,oversampling_rate,training):
    features1_tr = []
    features2_tr = []
    features3_tr = []
    target1_tr = []
    target2_tr = []
    target3_tr = []
    target31_tr = []
    target_inj_tr = []

    for i in training:
        for j in range(all_pressures.shape[1]):
            for k in range(all_pressures.shape[2]):
                for l in range(all_pressures.shape[3]):
                    for m in range(all_pressures.shape[4]):
                        if (k,l) not in [(13,13),(23,23)]:
                            features1_tr.append([k/24,l/24,m/2,j/71,all_permeabilities[i][j][k][l][m],all_porosities[i][j][k][l][m],all_surf_inj_rate_series[i][j]])
                            features2_tr.append([j/71,Ks[i]/2,Rs[i]/7])

                            features3_tr.append([[all_permeabilities[i][j//10][k//10][l//10][0],0,0],[0,all_permeabilities[i][j//10][k//10][l//10][1],0],[0,0,all_permeabilities[i][j//10][k//10][l//10][2]]])
                            target1_tr.append(all_saturations[i][j][k][l][m])
                            target2_tr.append(all_pressures[i][j][k][l][m])
                            target3_tr.append(all_surf_prod_rate_series[i][j])
                            target31_tr.append(0)
                            target_inj_tr.append(0)
                        else:
                            features1_tr+=[[k/24,l/24,m/2,j/71,all_permeabilities[i][j][k][l][m],all_porosities[i][j][k][l][m],all_surf_inj_rate_series[i][j]]] * oversampling_rate

                            features2_tr+=[[j/71,Ks[i]/2,Rs[i]/7]] * oversampling_rate

                            features3_tr+=[[[all_permeabilities[i][j//10][k//10][l//10][0],0,0],[0,all_permeabilities[i][j//10][k//10][l//10][1],0],[0,0,all_permeabilities[i][j//10][k//10][l//10][2]]]] * oversampling_rate
                            target1_tr += [all_saturations[i][j][k][l][m]] * oversampling_rate
                            target2_tr += [all_pressures[i][j][k][l][m]] * oversampling_rate
                            target3_tr += [all_surf_prod_rate_series[i][j]] * oversampling_rate
                            target31_tr += [all_surf_prod_rate_series[i][j]] * oversampling_rate
                            target_inj_tr += [all_surf_inj_rate_series[i][j]] * oversampling_rate


    features1_tr = np.array(features1_tr)
    features2_tr = np.array(features2_tr,dtype='float32')
    features3_tr = np.array(features3_tr)
    target1_tr = np.array(target1_tr)
    target1_tr = np.expand_dims(target1_tr, axis=1)
    target2_tr = np.array(target2_tr)
    target2_tr = np.expand_dims(target2_tr, axis=1)
    target3_tr = np.array(target3_tr)
    target3_tr = np.expand_dims(target3_tr, axis=1)
    target31_tr = np.array(target31_tr)
    target31_tr = np.expand_dims(target31_tr, axis=1)
    target_inj_tr = np.array(target_inj_tr)
    target_inj_tr = np.expand_dims(target_inj_tr, axis=1)
    print(features1_tr.shape)
    print(features2_tr.shape)
    print(features3_tr.shape)
    print(target1_tr.shape)
    print(target2_tr.shape)
    print(target3_tr.shape)
    print(target31_tr.shape)
    print(target_inj_tr.shape)
    
    return features1_tr,features2_tr,features3_tr,target1_tr,target2_tr,target3_tr,target31_tr,target_inj_tr

    
    
