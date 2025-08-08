for s in 1 2 3 4 5
do 
    for ss in 0 1 2 3 4
    do
        python -m model.gwave.load_wavenet_dsa -de ${ss} -d ../data/CA -lrs 1e-3 -e 100 -lp results/dc_dsa_cluster_ca_1e-3_1e-3_${s}.pt -sp results_txt/ours_ca_${s}.txt -s ${ss} &
    done
    wait
done


for s in 0 1 2 3 4
do 
    for ss in 0 1 2 3 4
    do
        python -m model.gwave.load_wavenet_baseline -de ${ss} -d ../data/CA -lrs 1e-3 -lp results/random_ca_${s}.pt -sp results_txt/random_ca_${s} -e 100 -s ${ss} &
    done
    wait
done

for s in 1 2 3 4 5
do 
    for ss in 0 1 2 3 4
    do
        python -m model.gwave.load_wavenet_baseline -de ${ss} -d ../data/CA -lrs 1e-3 -lp results/dc_ca_1e-2_1e-2_${s}.pt -sp results_txt/dc_ca_${s} -e 100 -s ${ss} &
    done
    wait
done

for s in 0 1 2 3 4
do 
    for ss in 0 1 2 3 4
    do
        python -m model.gwave.load_wavenet_baseline -de ${ss} -d ../data/CA -lrs 1e-3 -lp results/condtsf_ca_${s}.pt -sp results_txt/condtsf_ca_${s} -e 100 -s ${ss} &
    done
    wait
done

for s in 0 1 2 3 4
do 
    for ss in 0 1 2 3 4
    do
        python -m model.gwave.load_wavenet_baseline -de ${ss} -d ../data/CA -lrs 1e-3 -lp results/kcenter_ca_${s}.pt -sp results_txt/kcenter_ca_${s} -e 100 -s ${ss} &
    done
    wait
done

for s in 1 2 3 4 5
do 
    for ss in 0 1 2 3 4
    do
        python -m model.gwave.load_wavenet_baseline -de ${ss} -d ../data/CA -lrs 1e-3 -lp results/dm_ca_1e-4_1e-2_${s}.pt -sp results_txt/dm_ca_${s} -e 100 -s ${ss} &
    done
    wait
done

for s in 0 1 2 3 4
do 
    for ss in 0 1 2 3 4
    do
        python -m model.gwave.load_wavenet_baseline -de ${ss} -d ../data/CA -lrs 1e-3 -lp results/herding_ca_${s}.pt -sp results_txt/herding_ca_${s} -e 100 -s ${ss} &
    done
    wait
done