package com.prog.conc.td1.app;

import com.prog.conc.td1.app.Memory;
import com.prog.conc.td1.app.Transaction;
import com.prog.conc.td1.app.TransactionAbort;

public class Main extends Thread {

    int type = 0;

    public Main() {}
    public Main(int type) {
        this.type = type;
    }

    public int delay(int n) {
        try {
            if(n == 0)
                n = 16;
            Thread.sleep(1 + (int)(n * Math.random()));
        } catch(InterruptedException e) {}

        return n < 512 ? n << 1 : n;
    }

    public void test() {
        int n = 0;
        try {
            Transaction transaction = Transaction.Transaction.get();

            transaction.begin();

            int val = transaction.read(0);
            transaction.write(0, val + 1);

            transaction.commit();

            System.out.println("---> " + Memory.memory.values[0].value);
        } catch(TransactionAbort abort) {
            // System.out.println("---> Abort");
            n = delay(n);
            test();
        }
    }

    public void test2() {
        int n = 0;
        try {
            Transaction transaction = Transaction.Transaction.get();

            transaction.begin();

            int t1 = transaction.read(0);
            int t2 = transaction.read(1);
            int lastP = transaction.read(2);
            int newP = lastP + (1 / (t1 - t2));

            transaction.write(2, newP);

            transaction.commit();

            System.out.println("---> " + Memory.memory.values[0].value);
        } catch(TransactionAbort abort) {
            // System.out.println("---> Abort");
            n = delay(n);
            test2();
        }
    }

    public void test2Prime() {
        int n = 0;
        try {
            Transaction transaction = Transaction.Transaction.get();

            transaction.begin();

            transaction.write(0, 217);
            transaction.write(1, 4);

            transaction.commit();

            System.out.println("---> " + Memory.memory.values[0].value);
        } catch(TransactionAbort abort) {
            // System.out.println("---> Abort");
            n = delay(n);
            test2Prime();
        }
    }

    public synchronized void testSync() {
        Memory.memory.values[0].value++;
    }

    public void run() {
        for(int i=0; i<10000; i++) {
            if (this.type == 0) {
                test2();
            } else {
                test2Prime();
            }
        }
    }

    public static void main(String args[]) throws Exception {

        Memory.memory.values[0].value = 4;
        Memory.memory.values[1].value = 5;


        int n = 99;
        Thread threads[] = new Thread[n];

        for(int i=0; i<n; i++) {
            if (i % 2 == 0) {
                (threads[i] = new Main(0)).start();
            } else {
                (threads[i] = new Main(1)).start();
            }
        }


        (new Main()).run();

        for(int i=0; i<n; i++)
            threads[i].join();

        System.out.println("============ RESULTS ============");
        System.out.println("Value in memory[0] : " + Memory.memory.values[0].value);
        System.out.println("Value in memory[1] : " + Memory.memory.values[1].value);
        System.out.println("Value in memory[2] : " + Memory.memory.values[2].value);
        System.out.println("Clock Globale de Memory : " + Memory.memory.clock);
        System.out.println("Number of successful commit : " + Transaction.commitSuccess);
        System.out.println("Number of aborted commit : " + Transaction.commitAbort);

    }
}
