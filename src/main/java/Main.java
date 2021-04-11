import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.math3.stat.regression.SimpleRegression;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException {

        String inputFilePath = "1min_ETHUSDT.csv";
        if(args.length != 0){
            inputFilePath = args[0];
        }

        System.out.println("Input file: " + inputFilePath);

        List<Double> closePrices = readCloseFromCSV(Path.of(inputFilePath));

        HurstResult result = hurst(closePrices.stream().mapToDouble(Double::doubleValue).toArray());
        System.out.println(result);
    }

    private static List<Double> readCloseFromCSV(Path filePath) throws IOException {
        List<Double> closePrices = new ArrayList<>();
        try (
                Reader reader = Files.newBufferedReader(filePath);
                CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader());
        ) {
            for (CSVRecord csvRecord : csvParser) {
                String close = csvRecord.get(4);
                closePrices.add(Double.parseDouble(close));
            }
        }

        return closePrices;
    }

    private static double[] toPct(double[] series){
        // series is longer than pcts by 1
        double[] pcts = new double[series.length-1];
        for(int i=0; i<pcts.length; i+=1){
            pcts[i] = series[i+1]/series[i] - 1;
        }

        return pcts;
    }

    public static class HurstResult {
        public double h;
        double c;
        int[] windowSizes;
        double[] rsMeans;

        public HurstResult(double h, double c, int[] windowSizes, double[] rsMeans) {
            this.h = h;
            this.c = c;
            this.windowSizes = windowSizes;
            this.rsMeans = rsMeans;
        }

        @Override
        public String toString(){
            return "H: " + h + ", c: " + c + ", \nwindow sizes: " + Arrays.toString(windowSizes)
                    +", RS: " + Arrays.toString(rsMeans);
        }
    }

    private static HurstResult hurst(double[] series){
        int minWindow= 10;
        int maxWindow= series.length - 1;

        if(series.length < 100){
            throw new IllegalArgumentException("Series length must be greater or equal to 100");
        }

        double log1 = Math.log10(minWindow);
        double log2 = Math.log10(maxWindow);
        double nElemsDouble = (log2-log1)/0.25 + 1; //+1 as first element is always there (aka. we start from index 0)
        int nElems = (int) Math.floor(nElemsDouble);
        // Substract 1 from array size if division (log2-log1)/0.25 is integer (to create open range - excluding last element)

        //math.log10(min_window), math.log10(max_window)/0.25 is THE SAME

        if(nElemsDouble == nElems){
            nElems -= 1;
            if(nElems < 0){
                throw new RuntimeException("Window sizes range size would be zero");
            }
        }

        int[] windowSizes = new int[nElems + 1]; //+1 to make space for length
        for(int i=0; i< windowSizes.length - 1; i+=1) {
            double val = log1 + i * 0.25;
            windowSizes[i] = (int) Math.floor(Math.pow(10, val));
        }

        windowSizes[windowSizes.length - 1] = series.length;


        List<Double> rsMeans = new ArrayList<>();
        for(int w: windowSizes){
            List<Double> rs = new ArrayList<>();
            for(int i=0; i<series.length; i+=1){
                int start = i * w;
                if(start + w > series.length){
                    break;
                }

                double val = simplifiedRs(Arrays.copyOfRange(series, start, start + w));

                if(val != 0){
                    rs.add(val);
                }
            }

            rsMeans.add(mean(rs));
        }

        // First 'column' is 'log10(window_sizes)', second 'log10(rsMeans)' - for each elements of those lists/arrays
        double[][] inp = new double[rsMeans.size()][2];

        for(int i=0; i < windowSizes.length; i+=1){
            inp[i][0] = Math.log10(windowSizes[i]);
            inp[i][1] = Math.log10(rsMeans.get(i));
        }

        SimpleRegression simpleRegression = new SimpleRegression();
        simpleRegression.addData(inp);

        double h = simpleRegression.getSlope();
        double c = simpleRegression.getIntercept();

        c = Math.pow(10, c);
        // H, c, [window_sizes, RS]
        double[] rsMeansArr = new double[rsMeans.size()];
        for(int i=0; i<rsMeansArr.length; i+=1){
            rsMeansArr[i] = rsMeans.get(i);
        }

        return new HurstResult(h, c, windowSizes, rsMeansArr);
    }

    private static double standardDeviation(double[] series, int deltaDegreesOfFreedom){
        double standardDeviation = 0;
        double sum = 0;
        for (int i = 0; i < series.length; i++) {
            sum = sum + series[i];
        }

        double mean = sum / series.length;

        for (int i = 0; i < series.length; i++) {
            standardDeviation = standardDeviation + Math.pow((series[i] - mean), 2);
        }

        if(series.length - deltaDegreesOfFreedom <= 0){
            throw new RuntimeException("Can't set " + deltaDegreesOfFreedom + " degrees of freedom for series of length " + series.length);
        }

        double sq = standardDeviation / (series.length - deltaDegreesOfFreedom);
        return Math.sqrt(sq);
    }

    private static double simplifiedRs(double[] series){
        double[] pcts = toPct(series);
        double min = Arrays.stream(series).min().getAsDouble();
        double max = Arrays.stream(series).max().getAsDouble();
        double R = max / min - 1; // range in percent
        double S = standardDeviation(pcts, 1); // np.std(pcts, ddof=1)

        if(R==0 || S==0){
            return 0;
        }

        return R / S;
    }

    public static double mean(List<Double> m) {
        double sum = 0;
        for (int i = 0; i < m.size(); i++) {
            sum += m.get(i);
        }
        return sum / m.size();
    }
}
