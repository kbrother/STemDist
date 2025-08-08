lr_arr=(1e-2 1e-3 1e-4)
for i in 0 1 2
do
    python -m model.gwave.train_wavenet -de ${i} -d ../data/AURORA -lr ${lr_arr[$i]} -e 100 -sp results_txt/real_aurora_${lr_arr[$i]}.txt -b 32 -s 0 &
done

for i in 0 1 2 3
do 
    python -m model.gwave.train_wavenet -de $(($i + 3)) -d ../data/AURORA -lr 1e-2 -e 100 -sp results_txt/real_aurora_1e-2_$(($i+1)).txt -b 32 -s $(($i+1)) &
done
wait