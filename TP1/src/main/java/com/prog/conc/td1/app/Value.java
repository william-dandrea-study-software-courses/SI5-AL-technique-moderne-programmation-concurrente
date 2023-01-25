package com.prog.conc.td1.app;

public class Value {
    public Integer value;
    public int counter;

    public Value(int value, int counter) {
        this.value = value;
        this.counter = counter;
    }

    @Override
    public String toString() {
        return "Value : { value=" + value + ", counter=" + counter + '}';
    }
}
