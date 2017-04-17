"""
sumarize.py
"""
import sys
import pickle

def main():
    """
    Number of users collected:
    Number of messages collected:
    Number of communities discovered:
    Average number of users per community:
    Number of instances per class found: This part will have a pos:number, and neg:number
    One example from each class:   This part will have a pos or neg and text
    4 is Positvie 0 Negative
    """
    f = open('./data/sum.txt', 'rb')
    fo = open('./summary.txt', 'w')
    num_of_users_message = pickle.load(f)
    fo.write('Number of users collected: ' + str(num_of_users_message[0]) + '\n')
    fo.write('Number of messages collected: ' + str(num_of_users_message[1]) + '\n')

    num_of_communities_percom = pickle.load(f)
    fo.write('Number of communities discovered: ' + str(num_of_communities_percom[0]) + '\n')
    fo.write('Average number of users per community: ' + str(num_of_communities_percom[1]) + '\n')

    classify = pickle.load(f)
    fo.write('Number of instances per class found:' + '\n')
    fo.write('Positive: ' + str(classify[0]) + '  ' + 'Negative: ' + str(classify[1]) + '\n')
    fo.write('One example from each class: ' + '\n')
    if classify[2] == 4:
        fo.write('Result is Positive' + '\n')
    else:
        fo.write('Result is Negative' + '\n')
    fo.write('Message: ' + str(classify[3]) )

if __name__ == '__main__':
    main()
