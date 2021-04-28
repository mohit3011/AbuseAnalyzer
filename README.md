# AbuseAnalyzer
Repository for our paper "AbuseAnalyzer: Abuse Detection, Severity and Target Prediction for Gab Posts" (Accepted at COLING 2020). This repository contains the code, data and other scripts used for the project.

**For additional information/queries please contact me via email (mohit.chandra@research.iiit.ac.in)**

## Dataset Information

### Encodings for Hate/Non-Hate labels
* **1** : Non-Hateful
* **2** : Hateful

### Encodings for Target of Hate labels
* **1-2** : Individual Second Person
* **1-3** : Individual Third Person
* **2** : Group

### Encodings for Class of Hate labels
* **1** : Biased Attitude
* **2** : Act of Bias and Discrimination
* **3** : Violence and Genocide

**Note** : The usermentions in the dataset have been changed to `@usermention` for privacy and ethical reasons.

## Citation

* If you use/refer the dataset and/or code presented in this paper, then kindly cite our work using the following BibTeX:

```
@inproceedings{chandra-etal-2020-abuseanalyzer,
    title = "{A}buse{A}nalyzer: Abuse Detection, Severity and Target Prediction for Gab Posts",
    author = "Chandra, Mohit  and
      Pathak, Ashwin  and
      Dutta, Eesha  and
      Jain, Paryul  and
      Gupta, Manish  and
      Shrivastava, Manish  and
      Kumaraguru, Ponnurangam",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.552",
    doi = "10.18653/v1/2020.coling-main.552",
    pages = "6277--6283",
    abstract = "While extensive popularity of online social media platforms has made information dissemination faster, it has also resulted in widespread online abuse of different types like hate speech, offensive language, sexist and racist opinions, etc. Detection and curtailment of such abusive content is critical for avoiding its psychological impact on victim communities, and thereby preventing hate crimes. Previous works have focused on classifying user posts into various forms of abusive behavior. But there has hardly been any focus on estimating the severity of abuse and the target. In this paper, we present a first of the kind dataset with 7,601 posts from Gab which looks at online abuse from the perspective of presence of abuse, severity and target of abusive behavior. We also propose a system to address these tasks, obtaining an accuracy of âˆ¼80{\%} for abuse presence, âˆ¼82{\%} for abuse target prediction, and âˆ¼65{\%} for abuse severity prediction.",
}
```


