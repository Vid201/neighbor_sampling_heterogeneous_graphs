train_runimp_mag240m_original:
	cd src ; python3 main.py --mode train --model runimp --dataset mag240m --hns original ; cd .. ;

test_runimp_mag240m_original:
	cd src ; python3 main.py --mode test --model runimp --dataset mag240m --hns original ; cd .. ;

calculate_hyperparameters:
	echo "calculate hyperparameters"