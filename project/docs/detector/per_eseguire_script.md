sys.path.append("src/detector")










PER ESEGUIRE SCRIPT IN ORDINE :

/////////////////////////////////////////////////////////////////////////////////////////////

1)
bfd_ngram_random_vs_real:

python3 src/detector/scripts/calcolo_metriche/bfd_ngram_random_vs_real.py \
  --root   data/detector/datasets \
  --out   data/detector/derived/vari_csv/csv_distanza_coppie_random_vs_real \
  --pairs 20000 \
  --ngram 2 \
  --buckets 65536




dentro --root si aspetta di trovare:
pdf data/PDF-total/
pdf data/pdf_ranflood/

txt data/TXT-total/
txt data/txt_ranflood/

jpg data/JPG-total/
jpg data/jpg_ranflood/

docx data/DOCX-total/
docx data/docx_ranflood/




/////////////////////////////////////////////////////////////////////////////////////////////


2)
print_summary_no_plots:

python3 src/detector/scripts/calcolo_metriche/print_summary_no_plots.py




/////////////////////////////////////////////////////////////////////////////////////////////

3)
plot_distance:

python3 src/detector/scripts/calcolo_metriche/plot_distance.py


/////////////////////////////////////////////////////////////////////////////////////////////

4)
variance_analysis:

python3 src/detector/scripts/calcolo_metriche/variance_analysis.py

i dati li prende da:
   "pdf":  ["pdf data/PDF-total"],
    "txt":  ["txt data/TXT-total"],
    "jpg":  ["jpg data/JPG-total"],
    "docx": ["docx data/DOCX-total"],


salva in data/detector/derived/vari_csv/csv_varianza/variance_from_mean_summary.csv
media
deviazione standard
varianza
percentile 95


/////////////////////////////////////////////////////////////////////////////////////////////
///in UTILS
5)
split_dataset_kfold:

python3 src/detector/data/kfold_split.py \
  --out data/detector/derived/vari_json/json_split_dataset/kfold_split_3.json \
  --k_folds 3 \
  --seed 42


x/////////////////////////////////////////////////////////////////////////////////////////////
//in OTTIMIZZAZIONE
6)
genera_Score_kfold:

python3 src/detector/scripts/generazione_soglie/generate_scores_kfold.py \
  --out data/detector/csv_utils/csv_train_e_test_tutti_i_fold 



/////////////////////////////////////////////////////////////////////////////////////////////

7)
optimizie_threshold_kfold:

python3 src/detector/scripts/generazione_soglie/optimize_thresholds_kfold.py \
  --scores_dir data/detector/csv_utils/csv_train_e_test_tutti_i_fold \
  --out_dir data/detector/csv_utils/soglie_ottimizzate_per_ogni_fold \
  --k_folds 3 \
  --alpha 1.0 \
  --beta 20.0




/////////////////////////////////////////////////////////////////////////////////////////////



8)
metrics_eval_plausibility_kfold

python3 src/detector/scripts/calcolo_metriche/metrics_eval_plausibility_kfold.py






/////////////////////////////////////////////////////////////////////////////////////////////



8.1)

  PER CALCOLARE FP/FV NEI MIEI DATASET CON SOGLIE STATISTICHE:

  PRIMA FAI:
   
          python3 src/detector/data/train_test_split.py \
    --out data/detector/csv_utils/json_split_dataset_per_un_fold/train_test_split.json

    


POI FAI:

  python3 src/detector/scripts/generazione_soglie/generate_scores.py \
  --out data/detector/csv_utils/csv_train_e_test_un_fold


POI FAI:

python3 src/detector/scripts/calcolo_metriche/metrics_eval_sigma.py 






/////////////////////////////////////////////////////////////////////////////////////////////


9)
compute_centroids

python3 src/detector/core/compute_centroids.py

salva un json con i centroidi dei vari formati in "data/detector/derived/vari_json/centroidi_ogni_formato/centroids.json"



/////////////////////////////////////////////////////////////////////////////////////////////

10)
compute_final_thresholds:

python3 src/detector/thresholds/compute_final_thresholds.py \
  --inputs data/detector/csv_utils/soglie_ottimizzate_per_ogni_fold/thresholds_all_formats_train_fold0.csv \
           data/detector/csv_utils/soglie_ottimizzate_per_ogni_fold/thresholds_all_formats_train_fold1.csv \
           data/detector/csv_utils/soglie_ottimizzate_per_ogni_fold/thresholds_all_formats_train_fold2.csv \
  --out_csv data/detector/derived/vari_csv/csv_soglie_finali/final_thresholds_mean.csv \
  --out_json data/detector/derived/vari_json/json_soglie_finali/final_thresholds_mean.json








/////////////////////////////////////////////////////////////////////////////////////////////

11)
check_dataset_with_optimize_model:

python3 src/detector/scripts/detector/check_dataset_with_optimized_model.py \
  --dataset "data/detector/datasets/pdf data/pdf_ranflood" 





/////////////////////////////////////////////////////////////////////////////////////////////

12)
print_metrics_dataset_from_variance


python3 src/detector/scripts/detector/print_metrics_dataset_from_variance.py --dataset <cartella>


stessa cosa del detector finale ma con soglie statistiche e non soglie ottimizzate
