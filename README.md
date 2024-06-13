# MMA: Benchmarking Multi-Modal Large Language Models in Ambiguity Contexts

1. You have to set GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY to use this limited-access models.
2. Download csv file from: https://docs.google.com/spreadsheets/d/1jW9ukvBIfUJBBxVNs27K3n38VzVYotYAPCP8q-W_XNs/edit?usp=sharing
3. Place your csv file in a pth
4. Download dataset from: https://drive.google.com/file/d/1UnyDcFWBeWrYQNLrfva4B7UaZT6GMaKU/view?usp=sharing
5. Place your dataset in a path
6. Install dependency with requirement files.
7. python main.py --model [Model Name] #modify the dataset path and csv path in line#46 #47
8. Find your output csv path after running main.py
9. python evaluation.py  #modify the dataset path and csv path in line#4, and modify the models names in line#7
10. Final Ambiguity Accuracy of model will be given as this order: "Adjective","Noun","Verb", "Attachment Ambiguity", "Coordination Ambiguity","Structural Ambiguity","Pragmatic Ambiguity","Idiom", "Lexical", "Syntactic", "Semantic", "Overall".
