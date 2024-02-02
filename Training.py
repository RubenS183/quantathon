import math
import os
import pickle
import neat
import glob
import pandas as pd

def main(genomes, config):
    # Get the list of all CSV files in the current directory
    csvFiles = glob.glob("**/*.csv")
    data = {}

    # Read the csv files and store them in a list of Pandas DataFrames
    for csvFile in csvFiles:
      data[csvFile] = pd.read_csv(csvFile)
    
    del csvFile

    stocks = []
    cash = []
    nets = []
    ge = []
    result = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        ge.append(g)
        cash.append([1000000, 0])
        stocks.append({})
        result.append({})

    run = True

    for i in range(len(data[csvFiles[0]])):
        for x, g in enumerate(ge):
            
            cash[x][1] = 0
            
            for stock in stocks[x]:
                for holding in stocks[x][stock]:
                    cash[x][1] += data[stock]['VWAP'][i] * holding[1]
                    
            # print(cash[x][0] + cash[x][1])
                    
            
            for stockName, df in data.items():
                result[x][stockName] = nets[x].activate((df['VWAP'][i], df['Volume'][i], df['EMA'][i]))[0]

            result[x] = dict(sorted(result[x].items(), key=lambda item: item[1]))

            #Selling stocks
            for stock, res in result[x].items():
                if res > 0:
                    continue
                    
                if stock in stocks[x]:
                    stocks[x][stock] = sorted(stocks[x][stock], key=lambda item: item[0])
                    totalShares = 0
                    
                    for holding in stocks[x][stock]:
                        totalShares += holding[1]
                    
                    temp = math.floor(totalShares * res * (-1))
                    while temp > 0:
                        boughtShares = stocks[x][stock][-1][1]
                        boughtValue = stocks[x][stock][-1][0]
                        if boughtShares > 0:
                            boughtShares -= 1
                            
                            cash[x][0] += data[stock]['VWAP'][i]
                            cash[x][1] -= data[stock]['VWAP'][i]
                            
                            g.fitness += ((data[stock]['VWAP'][i] - boughtValue) / boughtValue) * 100
                            
                            temp -= 1
                        else:
                            stocks[x][stock].pop()
                    
                        
                        
                        
            
            #Reversing the list to find the highest results first  
            result[x] = dict(sorted(result[x].items(), key=lambda item: item[1], reverse = True))
            
            #Buying stocks
            for stock, res in result[x].items():
                if res < 0:
                    continue
                
                currentStockValuation = 0
                totalValuation = cash[x][0] + cash[x][1]
                
                for name in stocks[x]:
                    if name == stock:
                        for thing in stocks[x][stock]:
                            currentStockValuation += data[stock]['VWAP'][i] * thing[1]
                
                shares = 10#math.floor(((0.1 * totalValuation * res) - currentStockValuation) / data[stock]['VWAP'][i])
                
                if shares > 0 and cash[x][0] > shares*data[stock]['VWAP'][i]:
                    if stock in stocks[x]:
                        stocks[x][stock].append([data[stock]['VWAP'][i],shares])
                        cash[x][0] -= shares*data[stock]['VWAP'][i]
                        cash[x][1] += shares*data[stock]['VWAP'][i]
                        
                    else:
                        stocks[x][stock] = [[data[stock]['VWAP'][i],shares]]
                        cash[x][0] -= shares*data[stock]['VWAP'][i]
                        cash[x][1] += shares*data[stock]['VWAP'][i]
                    
        if len(ge) <= 0:
            run = False
            break
    
                
                
    # while run:

    #     for x, g in enumerate(ge):
    #         # if cash[x] <= 0:
    #         #     g.fitness -= 100
    #         #     nets.pop(x)
    #         #     ge.pop(x)
    #         #     stocks.pop(x)
    #         #     cash.pop(x)

    #         else:
    #             if ships[x].available_asteroids:
    #                 asteroid = ships[x].available_asteroids[0]
    #                 output = nets[x].activate((asteroid.rect.center[0], asteroid.rect.center[1],
    #                                            asteroid.angle, ships[x].x, ships[x].y))

    #                 if output[0] > 0:
    #                     # g.fitness += ships[x].move_right()
    #                     ships[x].move_right()
    #                 elif output[0] < 0:
    #                     # g.fitness += ships[x].move_left()
    #                     ships[x].move_left()
    #                 if output[1] <= 0:
    #                     ships[x].shoot()

            if event.type == pygame.QUIT:
                g = []
                for gen in ge:
                    g.append(gen.fitness)

                dot = ge[g.index(max(g))]
                with open('neuralNetworkc', 'wb') as f:
                    pickle.dump(dot, f)
                quit()
                run = False


def run(config_path):
    """
    RUns neat and evolves the neural network as per the configurations
    finds the best nn and pickles it
    :param config_path: fiole path to configuration file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 500)
    with open('neuralNetwork1', 'wb') as f:
        pickle.dump(winner, f)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)