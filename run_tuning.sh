lr_list=(1e-2 1e-3 1e-4)
for s in 0 1 2
do
    for g in 0 1 2
    do
        python -m model.agcrn.load_agcrn_dsa -de ${g} -d ../data/GBA -lr ${lr_list[$g]} -e 400 -lp results/dc_dsa_cluster_gba_1e-3_1e-3_1.pt -sp results_txt/ours_gba_${lr_list[$g]}.txt -s ${s} &
    done    
    wait
done

for s in 0 1 2
do
    for g in 0 1 2
    do
        python -m model.agcrn.load_agcrn_baseline -de ${g} -d ../data/GBA -lr ${lr_list[$g]} -lp results/dc_gba_1e-2_1e-3_1.pt -sp results_txt/dc_gba_${lr_list[$g]}.txt -e 400 -s ${s} &
    done
    wait
done

for s in 0 1 2
do
    for g in 0 1 2
    do
        python -m model.agcrn.load_agcrn_baseline -de ${g} -d ../data/GBA -lr ${lr_list[$g]} -lp results/condtsf_gba_0.pt -sp results_txt/condtsf_gba_${lr_list[$g]}.txt -e 400 -s ${s} &
    done
    wait
done

for s in 0 1 2
do
    for g in 0 1 2
    do
        python -m model.agcrn.load_agcrn_baseline -de ${g} -d ../data/GBA -lr ${lr_list[$g]} -lp results/kcenter_gba_0.pt -sp results_txt/kcenter_gba_${lr_list[$g]}.txt -e 400 -s ${s} &
    done
    wait
done