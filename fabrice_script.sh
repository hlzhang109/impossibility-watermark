./continuation_attacker.sh 0 "4_evan_2_2_1" "4_evan_2_2" 4
./continuation_attacker.sh 1 "4_evan_2_3_1" "4_evan_2_3" 4
./completion_attacker.sh 2 1 "completion" 1
./completion_attacker.sh 3 1 "completion" 2
./completion_attacker.sh 4 2 "completion" 1
./completion_attacker.sh 5 2 "completion" 2
python -m attack attack_args.cuda=6  attack_args.watermarked_text_path="./inputs/round_4_outputs.csv" attack_args.watermarked_text_num=3 attack_args.save_name="4_evan_3_3.csv" &> "4_evan_3_3.txt"
python -m attack attack_args.cuda=7  attack_args.watermarked_text_path="./inputs/round_4_outputs.csv" attack_args.watermarked_text_num=3 attack_args.save_name="4_evan_3_4.csv" &> "4_evan_3_4.txt"