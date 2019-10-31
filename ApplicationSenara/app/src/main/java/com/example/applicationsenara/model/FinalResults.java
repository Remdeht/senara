package com.example.applicationsenara.model;

public class FinalResults {

    private String image_name;
    private String watermeter;
    private int pointer_value_1;
    private int pointer_value_2;
    private int pointer_value_3;
    private Boolean flag_pointervalue;
    private int tally_counter_1;
    private Boolean flag_tally_counter_1;
    private int tally_counter_2;
    private Boolean flag_tally_counter_2;
    private int tally_counter_3;
    private Boolean flag_tally_counter_3;
    private int tally_counter_4;
    private Boolean flag_tally_counter_4;
    private int tally_counter_5;
    private Boolean flag_tally_counter_5;
    private int tally_counter_6;
    private Boolean flag_tally_counter_6;

    public FinalResults(String image_name, String watermeter, int pointer_value_1, int pointer_value_2, int pointer_value_3, Boolean flag_pointervalue, int tally_counter_1, Boolean flag_tally_counter_1, int tally_counter_2, Boolean flag_tally_counter_2, int tally_counter_3, Boolean flag_tally_counter_3, int tally_counter_4, Boolean flag_tally_counter_4, int tally_counter_5, Boolean flag_tally_counter_5, int tally_counter_6, Boolean flag_tally_counter_6) {
        this.image_name = image_name;
        this.watermeter = watermeter;
        this.pointer_value_1 = pointer_value_1;
        this.pointer_value_2 = pointer_value_2;
        this.pointer_value_3 = pointer_value_3;
        this.flag_pointervalue = flag_pointervalue;
        this.tally_counter_1 = tally_counter_1;
        this.flag_tally_counter_1 = flag_tally_counter_1;
        this.tally_counter_2 = tally_counter_2;
        this.flag_tally_counter_2 = flag_tally_counter_2;
        this.tally_counter_3 = tally_counter_3;
        this.flag_tally_counter_3 = flag_tally_counter_3;
        this.tally_counter_4 = tally_counter_4;
        this.flag_tally_counter_4 = flag_tally_counter_4;
        this.tally_counter_5 = tally_counter_5;
        this.flag_tally_counter_5 = flag_tally_counter_5;
        this.tally_counter_6 = tally_counter_6;
        this.flag_tally_counter_6 = flag_tally_counter_6;
    }

    public Boolean getFlag_pointervalue() {
        return flag_pointervalue;
    }

    public Boolean getFlag_tally_counter_1() {
        return flag_tally_counter_1;
    }

    public Boolean getFlag_tally_counter_2() {
        return flag_tally_counter_2;
    }

    public Boolean getFlag_tally_counter_3() {
        return flag_tally_counter_3;
    }

    public Boolean getFlag_tally_counter_4() {
        return flag_tally_counter_4;
    }

    public Boolean getFlag_tally_counter_5() {
        return flag_tally_counter_5;
    }

    public Boolean getFlag_tally_counter_6() {
        return flag_tally_counter_6;
    }

    public String getImage_name() {
        return image_name;
    }

    public String getWatermeter() {
        return watermeter;
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

    public int getTally_counter_1() {
        return tally_counter_1;
    }

    public int getTally_counter_2() {
        return tally_counter_2;
    }

    public int getTally_counter_3() {
        return tally_counter_3;
    }

    public int getTally_counter_4() {
        return tally_counter_4;
    }

    public int getTally_counter_5() {
        return tally_counter_5;
    }

    public int getTally_counter_6() {
        return tally_counter_6;
    }
}
