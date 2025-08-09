for s in 0 1 2 3 4
do 
    for ss in 0 1 2 3 4
    do
        python -m model.agcrn.load_agcrn_dsa -de ${ss} -d ../data/GBA -lr 1e-3 -e 400 -lp results/dc_dsa_cluster_gba_1e-3_1e-3_$(($s + 1)).pt -sp results_txt/ours_gba_${s}.txt -s ${ss} &
    done
    wait
done


for s in 0 1 2 3 4
do 
    for ss in 0 1 2 3 4
    do
        python -m model.agcrn.load_agcrn_baseline -de ${ss} -d ../data/GBA -lr 1e-2 -lp results/dc_gba_1e-2_1e-3_$(($s + 1)).pt -sp results_txt/dc_gba_${s}.txt -e 400 -s ${ss} &
    done
    wait
done

for s in 0 1 2 3 4
do 
    for ss in 0 1 2 3 4
    do
        python -m model.agcrn.load_agcrn_baseline -de ${ss} -d ../data/GBA -lr 1e-2 -lp results/condtsf_gba_${s}.pt -sp results_txt/condtsf_gba_${s}.txt -e 400 -s ${ss} &
    done
    wait
done

for s in 0 1 2 3 4
do 
    for ss in 0 1 2 3 4
    do
        python -m model.agcrn.load_agcrn_baseline -de ${ss} -d ../data/GBA -lr 1e-2 -lp results/kcenter_gba_${s}.pt -sp results_txt/kcenter_gba_${s}.txt -e 400 -s ${ss} &
    done
    wait
done

