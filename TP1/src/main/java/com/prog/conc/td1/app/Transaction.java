package com.prog.conc.td1.app;

public class Transaction {
    private ValueSet writeSet;
    private ValueSet readSet;
    private int clock;

    public static int commitSuccess = 0;
    public static int commitAbort = 0;

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
        commitAbort++;
        throw new TransactionAbort();
    }

    public void begin() {
        this.clock = Memory.memory.clock;
    }

    public int read(int idx) throws TransactionAbort {
        // Si la Value se trouve dans le write set, on renvoie sa valeur.
        // C'est dans le write set si on l'a déjà modifié auparavant
        if (this.writeSet.get(idx) != null) {
            return this.writeSet.map[idx].value;
        }

        Value value = Memory.memory.values[idx];
        // On vérifie que le compteur de la Value est inférieur à l'horloge locale
        // Si le compteur de la value est supérieur au compteur local, cela veut dire qu'un autre thread à modifier cette valeur
        // et que l'on doit donc avorter la transaction.
        if (value.counter >= this.clock) {
            abort();
        }

        this.readSet.set(idx, value);
        return value.value;
    }

    public void write(int idx, int value) throws TransactionAbort {
        if (this.writeSet.get(idx) != null) {
            this.writeSet.map[idx].value = value;
        } else {
            this.writeSet.set(idx, new Value(value, 0));
        }
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

            readSet.clear();
            writeSet.clear();

            Memory.memory.clock++;
            commitSuccess++;
        }
    }
}
