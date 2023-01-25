package com.prog.conc.td1.app;

public class CheckTransactionResult {

    boolean status;
    int indexStart;

    public CheckTransactionResult(boolean status, int indexStart) {
        this.status = status;
        this.indexStart = indexStart;
    }
}
