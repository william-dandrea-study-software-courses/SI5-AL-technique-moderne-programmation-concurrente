# Etape 1.
Créez une classe Value qui représente une valeur de la mémoire transactionnelle. Celle-ci possède deux entiers, value et counter. Vous pouvez ajouter un ou des constructeurs ainsi qu'une fonction toString().

# Etape 2.

* Créez ensuite une classe `Memory` qui possède un tableau de `Value` ainsi qu'une horloge `clock` (i.e entier). 
* Ajoutez un constructeur qui permettra d'allouer une mémoire de taille quelconque. 
* Le tableau sera rempli de Value qui auront la valeur zéro dans leur compteur. 
* La classe Memory possèdera aussi un champ statique memory, qui sera une instance d'elle-même, avec une taille de 1024.

# Etape 3.
Créer une classe vide TransactionAbort, qui ne fera qu'étendre Exception.

# Etape 4.
Nous avons maintenant besoin d'une structure de données pour gérer les read sets et les write sets. Nous utiliserons pour cela la classe ValueSet suivante:

```java
class ValueSet {
    // Tableau contenant toutes les valeurs du read/write set
    public Value map[];
    // Tableau contenant les indices utilisés dans le tableau map.
    public int   elements[];
    // Nombre d'indices utilisés.
    public int   n_elements;
    
    // Constructeur
    public ValueSet(int max) {
        this.map = new Value[max];
        this.elements = new int[max];
    }

    // Récupérer la valeur à un indice.
    Value get(int idx) {
        // On renvoie juste la valeur à l'indice donné (sera peut-être null).
        return map[idx];
    }

    // Mettre une valeur à un indice.
    boolean set(int idx, Value val) {
        // Si l'index est déjà  utilisé, on renvoie true (val est ignoré).
        if(get(idx) != null)
            return true;

        // Sinon, on enregistre l'indice comme utilisé...
        elements[n_elements++] = idx;

        // ...et on met la valeur à cet indice.
        map[idx] = val;

        return false;
    }

    // Effacer le set.
    void clear() {
        // Seuls les indices utilisés pointent vers des valeurs non-nulles.
        for(int i=0; i < n_elements; i++)
            map[elements[i]] = null;
        n_elements = 0;
    }
}
```

# Etape 5.
Il est maintenant temps de commencer à implémenter la classe Transaction, qui constitue le coeur du STM. Vous pouvez utiliser le squelette suivant :

```java
class Transaction {
    private ValueSet writeSet;
    private ValueSet readSet;
    private int      clock;

    private Transaction() {
        int mem_size = Memory.memory.values.length;
        writeSet = new ValueSet(mem_size);
        readSet  = new ValueSet(mem_size);
    }

    public static ThreadLocal<Transaction> Transaction =
        new ThreadLocal<Transaction>() {
            protected synchronized Transaction initialValue() {
                return new Transaction();
            }
        };

    public void abort() throws TransactionAbort {
        // à implémenter...
    }

    public void begin() {
        // à implémenter...
    }

    public int read(int idx) throws TransactionAbort {
        // à implémenter...
    }

    public void write(int idx, int value) throws TransactionAbort {
        // à implémenter...
    }

    public void commit() throws TransactionAbort {
        synchronized(Memory.memory) {
            if(!readSet.checkRead(clock)) {
                readSet.clear();
                writeSet.clear();
                abort();
            }
            if(!writeSet.checkWrite(clock)) {
                writeSet.clear();
                abort();
            }
            Memory.memory.clock++;
        }
    }
}
```

La suite vous guide pour implémenter les différentes fonctions.

### Etape 5.1.
Commencez par implémenter les fonction abort() et begin(). La première ne fait que renvoyer une exception, et la deuxième, qui correspond au début d'une transaction, ne fait que recopier l'horloge globale dans l'horloge locale.
### Etape 5.2.
Implémentez la fonction read(). L'idée est la suivante : si la Value se trouve dans le write set, on renvoie sa valeur. Sinon, on vérifie que le compteur de la Value est inférieur à  l'horloge locale si oui, on l'ajoute au read set et on renvoie sa valeur sinon on avorte la transaction.

### Etape 5.3.
Implémentez la fonction write(). L'idée est la suivante : si la Value se trouve dans le write set, on met à jour son attribut value. Sinon, on ajoute la Value au write set et on met à  jour son attribut value.

# Etape 6.
Votre STM est presque entièrement implémentée ! Il ne reste plus qu'à  gérer le commit. Celui-ci est déjà  presque implémenté : il vous faut simplement ajouter deux fonctions checkRead() et checkWrite() à  ValueSet.

### Etape 6.1.
Implémentez la fonction checkRead(). L'idée est de parcourir les indices utilisés par le read set, et de renvoyer false (conflit) ssi le compteur de l'une des valeurs est supérieur ou égal à  l'horloge locale. Profitez en pour effacer le read set.

En étant astucieux, vous pouvez faire en sorte que la fonction clear() appelé dans le cas d'un abort n'aura plus qu'à parcourir le reste du read set. Si vous n'êtes pas astucieux, le read set sera parcouru deux fois.
### Etape 6.2.
Implémentez la fonction checkWrite(). L'idée est de parcourir les indices utilisés par le write set, et de renvoyer false si le compteur de l'une des valeurs est supérieur ou égal à l'horloge locale. Sinon, les valeurs du write set sont copiées dans la mémoire globale, avec la valeur de l'horloge globale. De la même manière que précédemment, dans cette deuxième boucle, vous effacerez le write set.


# Etape 7.
Ajoutez deux compteurs statiques qui compteront le nombre de commits et d'aborts d'une transaction.
# Etape 8.
Il est maintenant temps de tester votre programme... Vous pouvez pour cela utiliser le fichier Main.java disponible où 100 threads font en parallèle 10.000 fois la séquence suivante : tmp = memoire[0];memoire[0] = tmp+1.

Quel est le résultat attendu ?
Votre STM fonctionne-t-elle comme prévu ?

Main.java est assez simple. Saurez-vous implémenter un exemple plus compliquer ?
