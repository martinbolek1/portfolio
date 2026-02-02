Úvod a kontext projektu

Banky používajú, podobne ako všetky ostatné spoločnosti, marketingové kampane na podporu predaja finančných produktov (napr. termínované vklady, pôžičky). V praxi však veľká časť oslovení končí bez výsledku – kampaň stojí čas a peniaze, no neprinesie očakávaný nárast tržieb. Preto som si na pomoc zavolal dáta.

Tento projekt pracuje s datasetom od spoločnosti UCI Machine Learning Repository, ktorý obsahuje viac než 41 000 záznamov klientov portugalskej bankovej inštitúcie. Dáta sú z obdobia 2008 až 2010 a zachytávajú makroekonomické ukazovatele, demografické údaje o klientoch a ich zákazníckom správaní.
Link na dataset: https://archive.ics.uci.edu/dataset/222/bank+marketing

Hlavný cieľ

-   Predikčný model, ktorý odhadne, či klient uzavrie termínovaný vklad ešte pred marketingovou kampaňou.
-   Určiť najdôležitejšie premenné, ktoré ovplyvňujú uzatvorenie termínovaného vkladu.
-   Navrhnúť vhodnú segmentáciu klientov (klustering), aby bolo možné lepšie cieliť kampane na konkrétne skupiny.

Čiastkové ciele

-   Získať dáta, upraviť, doplniť, očistiť a nahrať do databázy, z ktorej budú ďalej používané.
-   Vykonať permutation_importance a výber najrelevantnejších premenných.
-   Vytvoriť, vyhodnotiť a porovnať viacero predikčných modelov.
-   Vizualizovať a interpretovať výsledky v Power BI a využiť ich pre získanie konkurenčnej výhody (odporúčania pre kampaň).

Použité skills

- PostreSQL, SQL, Python, Power BI, Pandas, Sklearn, SQLalchemy, Numpy, K-prototypes, Scaler, Logistic Regression, OneHotEncoder, ColumnTransformer, Permutation_importance, RandomForestClassifier, GridSearchCV, Models validation, Pipeline, DAX, Joblib and more...

Kapitola 1 - Príprava dát

- V prvej kapitole som pripravil dáta tak, aby boli použiteľné na modelovanie aj reporting.
- Načítal som dataset UCI Bank Marketing a spravil zálohu pre kontrolu zmien.
- Nasimuloval som 10 % chýbajúcich hodnôt v stĺpcoch age, campaign, loan a uložil masky na validáciu.
- age a campaign som doplnil cez KNN imputer (po škálovaní) a kvalitu doplnenia som overil metrikami MAE a RMSE.
- loan som doplnil ako klasifikačný problém cez logistickú regresiu s preprocessingom (škálovanie + one-hot encoding) a vyhodnotil confusion matrix + accuracy.
- Odstránil som duplicity a vytvoril 2 výstupy:
    - bank_report pre Power BI (pridal som feature new_customer z pdays)
    - bank_ml pre ML pipeline (kategórie doplnené na "unknown", cieľ y premapovaný na 0/1)
- Obe tabuľky som nahrial do PostgreSQL (bank_report, bank_ml) cez SQLAlchemy.

Kapitola 2 - Predikčné modely + výber najdôležitejších premenných

- Načítal som vyčistené dáta z PostgreSQL (bank_ml) a kvôli výkonu som spravil stratifikovaný sampling na ~5 000 riadkov, aby zostal rovnaký pomer tried v y.
- Vytvoril som 2 sady vstupov:
    - Personal (pre-campaign): demografia + statusy (age, job, marital, education, default, housing, loan)
    - Economic: makro ukazovatele (emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed)
- Postavil som RandomForest klasifikátor v Pipeline (OneHotEncoder pre kategórie) a doladil hyperparametre cez GridSearchCV (2 optimalizácie: na F1 a na accuracy).
- Výsledky som hodnotil cez confusion matrix + classification report a upravil rozhodovací threshold (cieľ: lepšie zachytiť “áno” triedu pri nevyvážených dátach).
- Na identifikovanie najdôležitejších premenných som použil permutation_importance – v “personal” modeli vyšli ako top faktory najmä job, age, education.
- Porovnal som model bez makra vs. model s makro premennými – s ekonomickými premennými bol model lepší, preto som finálny variant uložil ako bank_predict_y.joblib pre ďalšie použitie (Power BI / scoring).
- Ďalší krok projektu: klustering pre segmentáciu a cielenejšie kampane.

Kapitola 3 - Segmentácia klientov

- Z PostgreSQL (bank_ml) som načítal dáta a spravil stratifikovaný sample ~10 000 riadkov (nový seed), aby bol zachovaný pomer tried y.
- Pre segmentáciu som použil K-Prototypes clustering (mix numerických + kategóriálnych premenných), na vstupy iba “personal” premenné: age, job, marital, education, default, housing, loan. Výsledkom bolo 8 klastrov.
- Na každý klaster som aplikoval môj uložený predikčný model (z kapitoly 2) a vypočítal som pravdepodobnosť nákupu (y_predict) + pomer “kupujúcich” v klastri.
- Vybral som Top 3 klastre podľa najvyššieho podielu y_predict=1 a spravil ich stručný profil (medián veku + najčastejšie kategórie), aby bolo jasné, koho presne cieliť.
- Pre reporting som vytvoril súhrnnú tabuľku s metrikami: ratio, veľkosť segmentu (n), odhadovaný počet konverzií a conversion rate (porovnanie vs priemer datasetu).
- Výstupy som uložil späť do databázy:
    - bank_cluster (každý klient + cluster + predikcia + reálne y)
    - results (agregované výsledky po clustroch)

Kapitola 4 - Power BI

- Napojil som Power BI priamo na PostgreSQL a importol tabuľky bank_report, bank_cluster, results.
- V Power Query som spravil rýchle čistenie a prípravu dát pre vizualizácie:
    - typy stĺpcov (napr. na whole number tam, kde to dáva zmysel),
    - zjednotenie hodnôt v day_of_week,
    - vytvorenie vekových rozsahov cez Column from Examples (napr. 32–41, 50–59 …) pre jednoduchšie segmentácie v grafoch.
- V DAX som pridal pomocný stĺpec day_in_order cez SWITCH, aby sa dni v týždni zobrazovali správne (Mon→Sun) a dali sa korektne zoradiť v osiach grafov.
- V modeli dát som vytvoril relationship results → bank_cluster (1:* cez cluster, cross-filter podľa potreby), aby sa dali agregované metriky segmentov používať spolu s klientskymi dátami a filtrovať reporty jedným klikom.
- Vytvoril som reporty (grafy + KPI) a doladil použiteľnosť:
    - vlastné Measures v DAX,
    - Edit interactions medzi vizuálmi,
    - Buttons + page navigation,
    - základný styling (čitateľnosť, formátovanie čísel, konzistentný layout).
- Reporty:
    - Demographics Data - vizualizácie pre lepšie pochopenie akými zákazníkmi banka disponuje
    - Marketing Data - vizualizácie výsledkov z marketingovej kampane
    - Predictions Outcome - vizualizácie z kapitôl 2-3 + Drill Through (po kliknutí na cluster sa otvorí detail: n v klastri, % s TD, conversion rate, odhad “guaranteed” vkladov + profil segmentu job/marital, age range, housing/loan).

Záver:

- Postavil som end-to-end pipeline: CSV → čistenie/imputácia → PostgreSQL → ML model → clustering → Power BI reporty.
- Vytvoril som 2 dátové vrstvy: bank_report (reporting) a bank_ml (modelovanie), plus výstupy bank_cluster a results pre segmentáciu.
- Otestoval som kvalitu prípravy dát (validácia imputácie) a postavil Random Forest model s tuningom + interpretáciou cez permutation importance.
- Segmentoval som klientov pomocou K-Prototypes (8 klastrov) a určil TOP segmenty podľa predikovaného podielu klientov s termínovaným vkladom.
- Výsledky som premenil na použiteľné odporúčania v Power BI: management dashboardy + “data science” stránka + drill-through do detailu klastru.
- Klastre 6, 7 a 5 mali vyššiu konverziu než priemer datasetu, pričom Cluster 6 bol približne o 25 % efektívnejší – vhodný cieľ hlavne pre banky bez rozpočtu na masový marketing (aj keď ide o menší segment).
- Dataset je nevyvážený (väčšina je y=0) a kvôli výkonu som model trénoval len na ~8 % dát, takže výsledky treba brať ako praktický prototyp a pri plnom tréningu by sa dali ešte zlepšiť.
