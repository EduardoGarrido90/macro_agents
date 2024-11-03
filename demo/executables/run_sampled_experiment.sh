#!/bin/bash

rm -rf ../results/*

python3 ../main.py --max_actions 18 --min_prod 3 --num_competitors 5 --max_fixed_costs 19 --min_fixed_costs 1 --elasticity 1.15754030286298 --base_demand 2.9052692918406002 --prod_noise 0.037225146603427815 --storage_factor 2.0278423448668663 --brand_effect 0.4691907034397638 --max_subsidy 11 --max_price 244
