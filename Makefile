train_runimp_mag240m_original:
	timestamp=$$(date +%d.%m.%y-%H.%M.%S) ; cd src ; nohup python3 main.py --mode train --model runimp --dataset mag240m --hns original > ../logs/train_runimp_mag240m_original_$${timestamp}.log 2>&1 &

test_runimp_mag240m_original:
	timestamp=$$(date +%d.%m.%y-%H.%M.%S) ; cd src ; nohup python3 main.py --mode test --model runimp --dataset mag240m --hns original > ../logs/test_runimp_mag240m_original_$${timestamp}.log 2>&1 &

calculate_hyperparameters:
	echo "calculate hyperparameters"