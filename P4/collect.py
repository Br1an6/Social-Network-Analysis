"""
collect.py
"""
import sys
import time
from TwitterAPI import TwitterAPI
import pickle

consumer_key = 'VsAo207IoIRF5AASDllD3H7yE'
consumer_secret = '4bNlsfogEbneVQp1TOLMk1ZGnwjbcqNeN8apiintWoa7bYJrHA'
access_token = '3164289948-CA0b188o68fVkbWJxXjkX13FnTmoKBplRf0nGZp'
access_token_secret = 'lz1ZweTDeKjZiBihfKcs2JQA3W58TNJfElqIA3aHYunEY'


def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


def get_users(twitter):
    return  twitter.request('users/search', {'q':'Chicago Bears','count':10}).json()

def get_users_friend(twitter, screen_names):
    """
        return a dict of users friends
    """
    ret_dict = {}
    for screen_name in screen_names:
        list_friends = twitter.request('friends/ids', {'screen_name':screen_name, 'count':20}).json()
        ret_dict[screen_name] = list_friends
    
    return ret_dict

def get_tweets(twitter, screen_names, tweets_count):
    ret_dict = {}
    for screen_name in screen_names:
        list_tweets = twitter.request('statuses/user_timeline', {'screen_name':screen_name, 'count':tweets_count, "lang": "en"}).json()
        ret_dict[screen_name] = list_tweets
    
    return ret_dict

def get_num_of_friends(users, friends_dict):
    num = 0
    for u in users:
        screen_name = u['screen_name']
        num += len(friends_dict[screen_name]['ids'])
    return num

def main():
    twitter = get_twitter()
    users = get_users(twitter)
    f = open('./data/users.txt','wb')
    pickle.dump(users, f)
    users_list = [ n['screen_name'] for n in users]
    friend_dict = get_users_friend(twitter, users_list)
    f2 = open('./data/friends.txt','wb')
    pickle.dump(friend_dict, f2)
    tweets_count = 200
    tweets = get_tweets(twitter, users_list, tweets_count)
    f3 = open('./data/tweets.txt','wb')
    pickle.dump(tweets, f3)
    test_data = []
    for key, val in tweets.items():
        for t in val:
            test_data.append(t['text'])
    """"
    train_dict = twitter.request('search/tweets', {'q':'Chicago Bears','count':20, "lang": "en"}).json()
    f4 = open('./data/train_tweets.txt','wb')
    pickle.dump(train_dict, f4)
    """
    num_friend = get_num_of_friends(users, friend_dict)
    list_of_summarize = []
    list_of_summarize.append(len(users_list) + num_friend)
    list_of_summarize.append(len(test_data))
    f4 = open('./data/sum.txt','wb')
    pickle.dump(list_of_summarize, f4)
if __name__ == '__main__':
    main()
