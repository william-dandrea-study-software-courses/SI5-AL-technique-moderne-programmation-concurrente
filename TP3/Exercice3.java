import java.util.HashMap;
import java.util.Map;

public class Exercice3 {

    public static int THREADS = 4;
    public static int ITERATIONS = 100000000;
    volatile public static ThreadInfo[] infos = new ThreadInfo[4];

    static class ThreadInfo {
        volatile public long progress;      // avancement d'un thread
        public long result;                 // résultat d'un thread
        public int[] padding = new int[64];

        public void usePadding() {this.padding[0] = 0;}
    }


    public static Map<String, Long> xorshf96(long x, long y, long z) {
        long t;

        x ^= x << 16;
        x ^= x >>> 5;
        x ^= x << 1;

        t = x;
        x = y;
        y = z;
        z = t ^ x ^ y;

        Map<String, Long> map = new HashMap<String, Long>();
        map.put("x", x);
        map.put("y", y);
        map.put("z", Math.abs(z));

        return map;
    }

    public static void main(String[] args) throws InterruptedException {

        Thread[] threads = new Thread[THREADS];

        long startTime = System.currentTimeMillis();

        for (int threadId = 0; threadId < THREADS; threadId++) {
            int finalThreadId = threadId;

            infos[finalThreadId] = new ThreadInfo();

            threads[finalThreadId] = new Thread(() -> {

                long x = 0L;
                long y = 362436069L;
                long z = 521288629L;

                for (int j = 0; j < ITERATIONS; j++) {
                    Map<String, Long> result = xorshf96(x, y, z);
                    x = result.get("x");
                    y = result.get("y");
                    z = result.get("z");

                    long xorValue = result.get("z");
                    long modulo = xorValue % 2;
                    infos[finalThreadId].result += modulo;
                    infos[finalThreadId].progress += 1;
                    infos[finalThreadId].usePadding();
                }
            });
            threads[finalThreadId].start();
        }

        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
        }

        for (int i = 0; i < THREADS; i++) {
            System.out.println("result[" + i + "] = " + infos[i].result + " | Progress : " + infos[i].progress );
        }

        long endTime = System.currentTimeMillis();
        System.out.println("Execution time: " + (endTime - startTime) / 1000.0 + " seconds");
    }

}
