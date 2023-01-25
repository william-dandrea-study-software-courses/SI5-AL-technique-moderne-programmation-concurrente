package com.prog.conc.td1.app;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Memory {
    public Value[] values;
    public int clock;
    public static Memory memory = new Memory(1024);

    public Memory(int size) {
        this.clock = 1;
        this.values = new ArrayList<Value>(Collections.nCopies(size, new Value(0, 0))).toArray(new Value[0]);
    }
}
