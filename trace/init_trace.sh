mkdir -p full_trace
root=/mnt/lustre/wgao/workspace/analyze_trace/trace
cp $root/MLaaS/MLaaS.csv full_trace/MLaaS.csv
cp $root/HeliosData/data/Venus/cluster_log.csv full_trace/Helios.csv
cp $root/Philly/philly_full.csv full_trace/Philly.csv
cp $root/BMtrace/sacct_1024.txt full_trace/BM.csv
