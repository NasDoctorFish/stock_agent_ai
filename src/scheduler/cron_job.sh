# cron_job.sh
#!/bin/bash
cd /Users/jju/Documents/Stock ML/notebooks/src_test/update_data.py
/Users/juwon/miniconda3/envs/mlinvest/bin/python update_data.py >> logs/update.log 2>&1
