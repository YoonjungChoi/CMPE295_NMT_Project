# Create Translation Models with OpenNMT Toolkit 

## 1. Dataset Preprocessing

We collected from Netflix and other websites from scratch. You can check sources from 'Collected_Dataset_Information.pdf'

Since each data set has different characteristics, preprocessing must be performed.

Netflix's subtitle has some unsupported or unnecessary information:

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


TATOEBA data set looks alot, but its quality is not good enough to use.

**Idioms** dataset between English and Korean is not easy. 

We collected them from [Idioms site](https://www.theidioms.com/) , [800 idioms PDF](https://www.academia.edu/11281938/The_800_Most_Commonly_Used_Idioms_in_America), [Kaggle English Idioms](https://www.kaggle.com/code/bryanb/scraping-sayings-and-proverbs/notebook#PART-I:-Scraping-English-sayings)


