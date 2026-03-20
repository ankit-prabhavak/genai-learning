from nltk.tokenize import TweetTokenizer # pyright: ignore[reportMissingImports]
from nltk.stem import SnowballStemmer, PorterStemmer, WordNetLemmatizer # pyright: ignore[reportMissingImports]

# Object initialization
tt = TweetTokenizer() # tokenizer
ss = SnowballStemmer("english") #stemmer
ps = PorterStemmer() 
lm = WordNetLemmatizer() # lemmatizer

# function to return tokens
def GetTokens(text):
    tokens = tt.tokenize(text)
    return tokens
    

# function to perform stemming
def StemToken(text):
    tokens = GetTokens(text)
    print(f"Generated Tokens: ", tokens)
    print() # enter new line
    StemList = [ps.stem(t) for t in tokens]

    return StemList

def GetLemmatization(text):
    tokens = GetTokens(text)
    print(f"Generated Tokens: ", tokens)

    LemmatizedText = [lm.lemmatize(t, pos="a") for t in tokens]
    return LemmatizedText


# main driver function
def Run():
    # user input
    text = input("Enter Your Text Here: ")
    print() # new line
    
    # performed stemming
    # stemmedText = StemToken(text)
    # print(f"After Stemming: ",stemmedText)
    # print()
    
    print("------Results------\n")
    LemmatizedText = GetLemmatization(text)
    print(f"After Lemmatization: ",LemmatizedText)


# Run the program
if __name__ == "__main__":
    Run()


