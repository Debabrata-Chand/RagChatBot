import pandas as pd 
from sentence_transformers import SentenceTransformer
import chromadb
df= pd.read_csv(r"C:\Users\debabrata chand\OneDrive\Desktop\RAG Chatbot\StudentData.csv")
#print(df)

#print("No of students: ",len(df))

documents=[] #an empty list to store document
for index,row in df.iterrows():
    document = f"""
Student ID: {row['Student_ID']}
Name: {row['Name']}
Age : {row['Age']}
Gender: {row['Gender']}
Department: {row['Department']}
Year: {row['Year']}
CGPA: {row['CGPA']}
Skills: {row['Skills']}
Interest: {row['Interest']}
Email: {row['Email']}
"""
    documents.append(document)

#print("First student document: ")
#print(documents[0])



#print("embedding dimensions:",len(embedding))


model=SentenceTransformer("all-miniLM-L6-v2")
embeddings=model.encode(documents)
#print("No of doucments: ",len(documents))
#print("No of embeddings: ",len(embeddings))
#print("Embedding dimensions: ",len(embeddings[0]))

#print("\nFirst student doc: ")
#print(documents[0])

#print("\nFirst student's embedding: ")
#print(embeddings[0])

client=chromadb.Client()

#print("ChromaDB client created successfully!")

collection=client.create_collection(name="students")
collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),     #.tolist() is used to convert the numpy array to a list
    ids=df["Student_ID"].tolist()
)
print("Data stored")

while True:
    question=input("Ask your question: ")
    question_embedding=model.encode(question).tolist()
    results=collection.query(
        query_embeddings=[question_embedding],
        n_results=1
    )

    print("\nResults:")
    for result in results["documents"][0]:
        print(result)

    again=input("\nDo you want to ask another question? (yes/no): ")
    if again.lower() != "yes":
        break
    