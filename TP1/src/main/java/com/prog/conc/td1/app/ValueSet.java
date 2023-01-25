package com.prog.conc.td1.app;

import java.util.Arrays;


public class ValueSet {

    // Tableau contenant toutes les valeurs du read/write set
    public Value map[];

    // Tableau contenant les indices utilisés dans le tableau map.
    public int elements[];

    // Nombre d'indices utilisés.
    public int n_elements;

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
        // Si l'index est déjà utilisé, on renvoie true (val est ignoré).
        if (get(idx) != null)
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
        for (int i = 0; i < n_elements; i++)
            map[elements[i]] = null;
        n_elements = 0;
    }

    public boolean checkRead(int clock) {

        for (int i = 0; i < n_elements; i++) {
            if(Memory.memory.values[elements[i]].counter >= clock) {
                return false;
            }
        }

        return true;
    }

    public boolean checkWrite(int clock) {

        for (int i = 0; i < n_elements; i++) {
            if(Memory.memory.values[elements[i]].counter >= clock) {
                return false;
            }
        }
        for (int i = 0; i < n_elements; i++) {
            Memory.memory.values[elements[i]] = new Value(map[elements[i]].value, Memory.memory.clock);
        }

        return true;
    }
}
