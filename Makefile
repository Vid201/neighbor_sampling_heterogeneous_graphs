train_runimp_mag240m_original:
	timestamp=$$(date +%d.%m.%y-%H.%M.%S) ; cd src ; nohup python3 main.py --mode train --model runimp --dataset mag240m --hns original > ../logs/train_runimp_mag240m_original_$${timestamp}.log 2>&1 &

test_runimp_mag240m_original:
	timestamp=$$(date +%d.%m.%y-%H.%M.%S) ; cd src ; nohup python3 main.py --mode test --model runimp --dataset mag240m --hns original > ../logs/test_runimp_mag240m_original_$${timestamp}.log 2>&1 &

heterogeneous_random_walks_demo:
	timestamp=$$(date +%d.%m.%y-%H.%M.%S) ; cd src/heterogeneous_random_walks/build ; cmake .. ; make ; nohup python3 ../main.py --walks 50 --length 10 --demo > ../../../logs/heterogeneous_random_walks_demo_$${timestamp}.log 2>&1 &

heterogeneous_random_walks:
	timestamp=$$(date +%d.%m.%y-%H.%M.%S) ; cd src/heterogeneous_random_walks/build ; cmake .. ; make ; nohup python3 ../main.py --walks 1000000 --length 10 > ../../../logs/heterogeneous_random_walks_$${timestamp}.log 2>&1 &

train_runimp_mag240m_option1:
	timestamp=$$(date +%d.%m.%y-%H.%M.%S) ; cd src ; nohup python3 main.py --mode train --model runimp_option1 --dataset mag240m --hns option1 > ../logs/train_runimp_mag240m_option1_$${timestamp}.log 2>&1 &

test_runimp_mag240m_option1:
	timestamp=$$(date +%d.%m.%y-%H.%M.%S) ; cd src ; nohup python3 main.py --mode test --model runimp_option1 --dataset mag240m --hns option1 > ../logs/test_runimp_mag240m_option1_$${timestamp}.log 2>&1 &