# snpDetect — détection de SNP sans génome de référence

Projet réalisé à l'ENSIMAG dans le cadre de l'UE *Introduction à la bioinformatique*, par :

- **Tony PEROTTINO** — [kibypoyo.github.io](https://kibypoyo.github.io)
- **Thomas SOUCHET** — [thomas-souchet.fr](https://thomas-souchet.fr)

## Le sujet

Une souche de la bactérie pathogène *Salmonella enterica* a muté et est devenue résistante à la tétracycline, son antibiotique usuel. Il s'agit d'élucider en quoi elle a muté, en comparant son code génétique à celui d'une souche restée sensible.

Tout l'intérêt de ce programme est le type de données d'entrée. Ce code utilise la segmentation d'Illumina qui bruite les données à hauteur de 1% en moyenne. Ce pipeline permet de débruiter ces erreurs pour comparer deux souches de bactéries (et peut-être étendu à d'autres espèces) et en déterminer les mutations sous forme de SNPs.

## Pourquoi s'en servir

Cette méthode reposant sur les données d'Illumina, elle n'utilise ni génome de référence, ni assemblage, ni alignement. Deux fichiers de reads suffisent en entrée au format '.fna'. Le tri entre erreurs de séquençage et vraies différences repose sur une démarche mathématique. 

Le programme est prévu pour des jeux de données volumineux : il parallélise les tâches pour réduire le temps de calcul et limite les pics mémoire, même sur des fichiers de l'ordre du gigaoctet. Chaque mutation est rendue avec de quoi juger de sa fiabilité, et le rapport indique jusqu'à quel taux d'erreur de séquençage la méthode reste valable.

Les choix d'implémentation qui le permettent, et les mesures qui le vérifient, sont détaillés dans le rapport.

## Exécution

```bash
pip install -r requirements.txt
python3 snpDetect.py --help
```

Les conseils d'exécutions sont aussi dans l'annexe B du rapport, et l'intérêt des paramètres y sont plus clairement expliqués.