# Understanding Dataset

Subtitle has some unsupported or unnecessary information:

```
CLEAN : ERROR TYPE
1. - [XX] (XX) ‎ => it should be removed in sentences.
2. remove whole sentence if it includes ♪ 
3. korean doesnt have punctuations to show "end of sentence", 
but English has puctuations like ! . ? <= three punctuations can be used to concat sentences.
```


optional)
4. some cases has "...", which makes it difficult to concat sentences.
To remove :
```
dataframe[0] = dataframe[0].str.replace("\u2026", '', regex=False)
```
<img width="642" alt="Screen Shot 2023-02-17 at 12 26 54 PM" src="https://user-images.githubusercontent.com/20979517/219786533-b1ba4839-7053-4404-b46e-97ef49a1b22c.png">
