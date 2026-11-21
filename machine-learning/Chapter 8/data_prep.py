import pandas as pd


print("==" * 50)
print("==" * 50)
files = open('review/review.txt' , 'r')
text = files.readlines()
print("Loaded file ")
print("first example")
print(text[1:3])

Reviews = []

Overall = []

counter = 0 

for line in range(len(text)):
     sen = text[line].strip('\n')
     review =[sen[(sen.index("text")):(sen.index("author") - 4)]]
   
    

     rating = sen[sen.index("ratings")+10:(sen.index("title") - 3)].strip().replace('"','').split(",")
     for i in range(len(rating)):

          rating[i] = rating[i].strip().split(':')
          number = rating[i][1]
          if  '}' in number: 
               number = number.replace('}' , ' ')
          overall =  float((number.strip()))
          if overall == 1 or overall == 5: 
            Overall.append(overall)
            Reviews.append(review)
          else: 
              pass
          break
     
     if line % 100000 == 0 : 
        print("100 Rows saved")
        print("==" * 50)  
        print("==" * 50) 
        print()
        print(Reviews[line:line+2])
        print(Overall[line:line+2])



print("==" * 50)  
print("==" * 50) 
print(f"Reviews length: {len(Reviews)}")
print(f"Overall length: {len(Overall)}")
print("==" * 50)
print("==" * 50)

# Then create the DataFrame
df = pd.DataFrame({"Review": Reviews, "Rating": Overall})
df.to_csv('reviews_dataset.csv', index=False)
print(df.head())

df = pd.read_csv("reviews_dataset.csv")
df['Rating'] = df['Rating'].map(int)
df["Review"] = df["Review"].apply(lambda x: x.replace("text", " "))

print("Loaded dataset")
print("==" * 50)
print("==" * 50)