save_path=/data/yanfeizhang/SES/local_wiki_server/wiki-18-e5-index
python download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz