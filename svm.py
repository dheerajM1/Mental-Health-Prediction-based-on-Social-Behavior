from sklearn import svm
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV

def vocablist():

    stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

    tweets = []
    X = []
    y = []
    for line in open('tweets.txt').readlines():
        items = line.split(',')
        tweets.append([int(items[0]), items[1].lower().strip()])

    # Extract the vocabulary of keywords
    vocab = dict()
    for class_label, text in tweets:
        for term in text.split():
            term = term.lower()
            if len(term) > 2 and term not in stopwords:
                if vocab.has_key(term):
                    vocab[term] = vocab[term] + 1
                else:
                    vocab[term] = 1

    vocab = {term: freq for term, freq in vocab.items() if freq > 5} # Remove terms whose frequencies are less than a threshold (e.g., 10)
    sorted_vocab=sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    sorted10_vocab=sorted_vocab[0:30] # picked top 10 sorted keys

    vocab1=dict(sorted10_vocab)
    vocab = {term: idx for idx, (term, freq) in enumerate(vocab1.items())}  #Generate an id (starting from 0) for each term in vocab


    for class_label, text in tweets:
        x = [0] * len(vocab)
        terms = [term for term in text.split() if len(term) > 2]
        for term in terms:
            if vocab.has_key(term):
                x[vocab[term]] += 1
        y.append(class_label)
        X.append(x)


    svc = svm.SVC(kernel='linear')
    Cs = range(1, 10)   #for 10 folder-cross validation
    #Cs = range(1, 20)  #for 20 folder-cross validation
    clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), cv=10)
    clf.fit(X, y)

    tweets = []
    for line in open('unlabelled_tweets.txt').readlines():
        tweets.append(line)


    # Generate X for testing tweets
    X = []
    for text in tweets:
        x = [0] * len(vocab)
        terms = [term for term in text.split() if len(term) > 2]
        for term in terms:
            if vocab.has_key(term):
                x[vocab[term]] += 1
        X.append(x)
    y = clf.predict(X)   #predicitng class label

    hello = map(str, y) # toconvert into string
    a = 0 #initialize the loop
    file = open("predicted_tweetssvm.txt", "a+")   #new file opened
    for text in tweets:
        labelled_tweet= hello[a]+","+text #appending predicted class and tweet text
        file.write(labelled_tweet)   #writting teh labelling tweet into file.
        a += 1

if __name__ == '__main__':
    vocablist()
