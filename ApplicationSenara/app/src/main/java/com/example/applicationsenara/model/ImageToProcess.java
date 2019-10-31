package com.example.applicationsenara.model;

import java.util.ArrayList;

public class ImageToProcess {
    private boolean safely_executed;
    private String image_name;
    private String watermeter;
    private double latitude;
    private double longitude;
    private int pointer_value_1;
    private int pointer_value_2;
    private int pointer_value_3;
    private ArrayList<Integer> tally_counter_1;
    private ArrayList<Integer> tally_counter_2;
    private ArrayList<Integer> tally_counter_3;
    private ArrayList<Integer> tally_counter_4;
    private ArrayList<Integer> tally_counter_5;
    private ArrayList<Integer> tally_counter_6;

    public ImageToProcess(boolean safely_executed, String image_name, String watermeter, int pointer_value_1, int pointer_value_2, int pointer_value_3, ArrayList<Integer> tally_counter_1, ArrayList<Integer> tally_counter_2, ArrayList<Integer> tally_counter_3, ArrayList<Integer> tally_counter_4, ArrayList<Integer> tally_counter_5, ArrayList<Integer> tally_counter_6) {
        this.safely_executed = safely_executed;
        this.image_name = image_name;
        this.watermeter = watermeter;
        this.pointer_value_1 = pointer_value_1;
        this.pointer_value_2 = pointer_value_2;
        this.pointer_value_3 = pointer_value_3;
        this.tally_counter_1 = tally_counter_1;
        this.tally_counter_2 = tally_counter_2;
        this.tally_counter_3 = tally_counter_3;
        this.tally_counter_4 = tally_counter_4;
        this.tally_counter_5 = tally_counter_5;
        this.tally_counter_6 = tally_counter_6;
    }

    public boolean isSafely_executed() {
        return safely_executed;
    }

    public String getImageName() {
        return image_name;
    }

    public String getWatermeter() {
        return watermeter;
    }

    public double getLatitude() {
        return latitude;
    }

    public double getLongitude() {
        return longitude;
    }

    public int getPointer_value_1() {
        return pointer_value_1;
    }

    public int getPointer_value_2() {
        return pointer_value_2;
    }

    public int getPointer_value_3() {
        return pointer_value_3;
    }

    public ArrayList<Integer> getTally_counter_1() {
        return tally_counter_1;
    }

    public ArrayList<Integer> getTally_counter_2() {
        return tally_counter_2;
    }

    public ArrayList<Integer> getTally_counter_3() {
        return tally_counter_3;
    }

    public ArrayList<Integer> getTally_counter_4() {
        return tally_counter_4;
    }

    public ArrayList<Integer> getTally_counter_5() {
        return tally_counter_5;
    }

    public ArrayList<Integer> getTally_counter_6() {
        return tally_counter_6;
    }
}
