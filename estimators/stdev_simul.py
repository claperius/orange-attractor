#!/usr/bin/env python3

################################################################################
#
# Mikolaj Sitarz 2019
# Apache License 2.0
#
# Demonstration code for article https://orange-attractor.eu/?p=311
#
################################################################################


import json
import multiprocessing
import numpy as np

import config


def calc_sample(population, sample_size):
    'perform calculations for given sample_size for both estimators'
    
    biased = []
    unbiased = []
    for k in range(config.average_pool_size):

        if config.the_same_samples:

            # draw random indices
            indices = [np.random.randint(0, len(population)) for k in range(sample_size)]

            # get samples
            sample = population[indices]
            
            # calculate both estimators
            std_biased, std_unbiased = sample.std(), sample.std(ddof=1)

        else:
            
            # draw random indices
            indices1 = [np.random.randint(0, len(population)) for k in range(sample_size)]
            indices2 = [np.random.randint(0, len(population)) for k in range(sample_size)]

            # get samples
            sample1 = population[indices1]
            sample2 = population[indices2]

            # calculate both estimators
            std_biased = sample1.std()
            std_unbiased = sample2.std(ddof=1)
        

        
        biased.append(std_biased)
        unbiased.append(std_unbiased)
        
    return np.mean(biased), np.mean(unbiased)


def store_result(data):
    'save result in output json file'
    
    if config.the_same_samples:
        tss = 's'
    else:
        tss = 'd'
        
    fname = 'result-{}-{}-{}-{}-{}.json'.format(config.distribution, config.population_size, config.average_pool_size, config.seed, tss)
    print('-> {}'.format(fname))

    with open(fname, 'w') as f:
        json.dump(data, f)


def get_sample_sizes():
    'return list of all sample sizes'
    
    result = []
    for sample_size in config.sample_sizes:
        if type(sample_size) is int:
            result.append(sample_size)
        else:
            result.extend(np.arange(*sample_size, dtype='int64').tolist())


    # store stdev calculated for whole population at the end
    if result[-1] != config.population_size:
        result.append(config.population_size)
            
    return result


def draw_and_calc(fargs):
    'for given population and sample size, draw samples, and calculate average values for both estimators'
    
    population, sample_size = fargs
    print('sample_size: {}'.format(sample_size))
    return sample_size, calc_sample(population, sample_size)
        
    
        
def main():

    # initialize pseudorandom numbers generator
    np.random.seed(config.seed)

    data = {
        "config": {
            "population_size": config.population_size,
            "sample_sizes": config.sample_sizes,
            "average_pool_size": config.average_pool_size,
            "seed": config.seed,
            "distribution": config.distribution,
            "the_same_samples": config.the_same_samples
        },
        "comment": "order: biased, unbiased",
        "result": []
    }

    # draw population
    if config.distribution == 'uniform':
        population = np.random.random(config.population_size)
    elif config.distribution == 'gauss':
        population = np.random.randn(config.population_size)
    elif config.distribution == 'poisson':
        population = np.random.poisson(size=config.population_size)
    else:
        raise Exception('unknown distribution: {}'.format(config.distribution))


        
    sample_sizes = get_sample_sizes()
    fargs = [(population, k) for k in sample_sizes]


    # use all available CPU's
    ncpu = multiprocessing.cpu_count()
    with multiprocessing.Pool(ncpu) as p:
        result = p.map(draw_and_calc, fargs)


    data['result'] = result
    store_result(data)

    
if __name__ == '__main__':
    main()
