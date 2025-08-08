python -m model.gwave.train_wavenet -de 0 -d ../data/CA -lr 1e-2 -e 100 -sp results_txt/real_ca_1e-2.txt -b 32 -s 0 &
python -m model.gwave.train_wavenet -de 1 -d ../data/CA -lr 1e-3 -e 100 -sp results_txt/real_ca_1e-3.txt -b 32 -s 0 &
python -m model.gwave.train_wavenet -de 2 -d ../data/CA -lr 1e-4 -e 100 -sp results_txt/real_ca_1e-4.txt -b 32 -s 0 &
wait

python -m model.gwave.train_wavenet -de 0 -d ../data/CA -lr 1e-2 -e 100 -sp results_txt/real_ca_1e-2_2.txt -b 32 -s 1 &
python -m model.gwave.train_wavenet -de 1 -d ../data/CA -lr 1e-2 -e 100 -sp results_txt/real_ca_1e-2_3.txt -b 32 -s 2 &
python -m model.gwave.train_wavenet -de 2 -d ../data/CA -lr 1e-2 -e 100 -sp results_txt/real_ca_1e-2_4.txt -b 32 -s 3 &
python -m model.gwave.train_wavenet -de 3 -d ../data/CA -lr 1e-2 -e 100 -sp results_txt/real_ca_1e-2_5.txt -b 32 -s 4 &
wait