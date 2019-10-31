package com.example.applicationsenara;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.NumberPicker;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.applicationsenara.model.FinalResults;
import com.example.applicationsenara.remote.APIUtils;
import com.example.applicationsenara.remote.FileService;

import java.util.Arrays;
import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class ResultsActivity extends AppCompatActivity{

    private FileService fileService;

    private TextView watermeterTextView;
    private String imageName;
    private String watermeter;

    private NumberPicker tallyCounterValue_1;
    private NumberPicker tallyCounterValue_2;
    private NumberPicker tallyCounterValue_3;
    private NumberPicker tallyCounterValue_4;
    private NumberPicker tallyCounterValue_5;
    private NumberPicker tallyCounterValue_6;

    private TextView tallyCounterValue_1Fixed;
    private TextView tallyCounterValue_2Fixed;
    private TextView tallyCounterValue_3Fixed;
    private TextView tallyCounterValue_4Fixed;
    private TextView tallyCounterValue_5Fixed;
    private TextView tallyCounterValue_6Fixed;

    private NumberPicker pointerValue_1;
    private NumberPicker pointerValue_2;
    private NumberPicker pointerValue_3;

    private Boolean flag_pointervalue;
    private Boolean flag_tallyCounterValue_1;
    private Boolean flag_tallyCounterValue_2;
    private Boolean flag_tallyCounterValue_3;
    private Boolean flag_tallyCounterValue_4;
    private Boolean flag_tallyCounterValue_5;
    private Boolean flag_tallyCounterValue_6;

    private Button saveDataButton;
    private ImageButton helpButton;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.results);
        fileService = APIUtils.getFileService();

        tallyCounterValue_1 = findViewById(R.id.np_tally_1);
        tallyCounterValue_2 = findViewById(R.id.np_tally_2);
        tallyCounterValue_3 = findViewById(R.id.np_tally_3);
        tallyCounterValue_4 = findViewById(R.id.np_tally_4);
        tallyCounterValue_5 = findViewById(R.id.np_tally_5);
        tallyCounterValue_6 = findViewById(R.id.np_tally_6);

        tallyCounterValue_1Fixed = findViewById(R.id.tcTextview1);
        tallyCounterValue_2Fixed = findViewById(R.id.tcTextview2);
        tallyCounterValue_3Fixed = findViewById(R.id.tcTextview3);
        tallyCounterValue_4Fixed = findViewById(R.id.tcTextview4);
        tallyCounterValue_5Fixed = findViewById(R.id.tcTextview5);
        tallyCounterValue_6Fixed = findViewById(R.id.tcTextview6);

        pointerValue_1 = findViewById(R.id.pointerValue_1);
        pointerValue_2 = findViewById(R.id.pointerValue_2);
        pointerValue_3 = findViewById(R.id.pointerValue_3);

        List<NumberPicker> numberPickers = Arrays.asList(
                tallyCounterValue_1, tallyCounterValue_2, tallyCounterValue_3,
                tallyCounterValue_4, tallyCounterValue_5, tallyCounterValue_6,
                pointerValue_1, pointerValue_2, pointerValue_3);

        for (final NumberPicker numberPicker: numberPickers) {
            numberPicker.setMinValue(0);
            numberPicker.setMaxValue(9);
        }

        Bundle b = getIntent().getExtras();
        if(b != null)
            imageName = b.getString("name");
            watermeter = b.getString("watermeter");

            // Determines if the tally counter value is interactive or not based on the reliability of the prediction

            checkInteractivity(b.getIntegerArrayList("tc_1").get(1), b.getIntegerArrayList("tc_1").get(0),
                    tallyCounterValue_1Fixed, tallyCounterValue_1);
            checkInteractivity(b.getIntegerArrayList("tc_2").get(1), b.getIntegerArrayList("tc_2").get(0),
                    tallyCounterValue_2Fixed, tallyCounterValue_2);
            checkInteractivity(b.getIntegerArrayList("tc_3").get(1), b.getIntegerArrayList("tc_3").get(0),
                    tallyCounterValue_3Fixed, tallyCounterValue_3);
            checkInteractivity(b.getIntegerArrayList("tc_4").get(1), b.getIntegerArrayList("tc_4").get(0),
                    tallyCounterValue_4Fixed, tallyCounterValue_4);
            checkInteractivity(b.getIntegerArrayList("tc_5").get(1), b.getIntegerArrayList("tc_5").get(0),
                    tallyCounterValue_5Fixed, tallyCounterValue_5);
            checkInteractivity(b.getIntegerArrayList("tc_6").get(1), b.getIntegerArrayList("tc_6").get(0),
                    tallyCounterValue_6Fixed, tallyCounterValue_6);

            pointerValue_1.setValue(b.getInt("pointer_val_1"));
            pointerValue_2.setValue(b.getInt("pointer_val_2"));
            pointerValue_3.setValue(b.getInt("pointer_val_3"));

        watermeterTextView = findViewById(R.id.owner);
        watermeterTextView.setText(watermeter);
        watermeterTextView.bringToFront();

        saveDataButton = findViewById(R.id.saveDataButton);
        saveDataButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Bundle bundle = getIntent().getExtras();
                updateValues(bundle);
            }
        });

        helpButton = (ImageButton) findViewById(R.id.helpButton);
        helpButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openInstructionsActivity();
            }
        });
    }

    public void openDataStoredActivity() {
        Intent intent = new Intent(ResultsActivity.this, DataStoredActivity.class);
        startActivity(intent);
    }

    public void openInstructionsActivity(){
        Intent intent = new Intent(ResultsActivity.this, InstructionsActivity.class);
        startActivity(intent);
    }

    public void checkInteractivity(Integer flag, Integer value, TextView textView, NumberPicker numberPicker){
        if (flag == 0) {
            textView.setText(value.toString());
            textView.setVisibility(View.VISIBLE);
            numberPicker.setVisibility(View.INVISIBLE);
            numberPicker.setEnabled(false);
        } else {
            numberPicker.setValue(value);
        }
    }

    @Override
    public void onBackPressed() {
        openCameraActivity();
    }

    public void openCameraActivity() {
        Intent intent = new Intent(this, CameraActivity.class);
        startActivity(intent);
    }

    private void updateValues(Bundle bundle) {
        // Checks if the user has changed any values and a boolean flag if this is the case

        if (pointerValue_1.getValue() != bundle.getInt("pointer_val_1") ||
                pointerValue_2.getValue() != bundle.getInt("pointer_val_2") ||
                pointerValue_3.getValue() != bundle.getInt("pointer_val_3")){
            flag_pointervalue = Boolean.TRUE;
        } else{
            flag_pointervalue = Boolean.FALSE;
        }

        if (tallyCounterValue_1 != null){
            if(tallyCounterValue_1.getValue() != bundle.getIntegerArrayList("tc_1").get(0)){
                flag_tallyCounterValue_1 = Boolean.TRUE;
            } else{
                flag_tallyCounterValue_1 = Boolean.FALSE;
            }
        } else{
            flag_tallyCounterValue_1 = Boolean.FALSE;
        }

        if (tallyCounterValue_2 != null){
            if(tallyCounterValue_2.getValue() != bundle.getIntegerArrayList("tc_2").get(0)){
                flag_tallyCounterValue_2 = Boolean.TRUE;
            } else{
                flag_tallyCounterValue_2 = Boolean.FALSE;
            }
        } else{
            flag_tallyCounterValue_2 = Boolean.FALSE;
        }

        if (tallyCounterValue_3!= null){
            if(tallyCounterValue_1.getValue() != bundle.getIntegerArrayList("tc_3").get(0)){
                flag_tallyCounterValue_3= Boolean.TRUE;
            } else{
                flag_tallyCounterValue_3= Boolean.FALSE;
            }
        } else{
            flag_tallyCounterValue_3= Boolean.FALSE;
        }

        if (tallyCounterValue_4 != null){
            if(tallyCounterValue_4.getValue() != bundle.getIntegerArrayList("tc_4").get(0)){
                flag_tallyCounterValue_4 = Boolean.TRUE;
            } else{
                flag_tallyCounterValue_4 = Boolean.FALSE;
            }
        } else{
            flag_tallyCounterValue_4 = Boolean.FALSE;
        }

        if (tallyCounterValue_5 != null){
            if(tallyCounterValue_5.getValue() != bundle.getIntegerArrayList("tc_5").get(0)){
                flag_tallyCounterValue_5 = Boolean.TRUE;
            } else{
                flag_tallyCounterValue_5 = Boolean.FALSE;
            }
        } else{
            flag_tallyCounterValue_5 = Boolean.FALSE;
        }

        if (tallyCounterValue_6 != null){
            if(tallyCounterValue_6.getValue() != bundle.getIntegerArrayList("tc_6").get(0)){
                flag_tallyCounterValue_6 = Boolean.TRUE;
            } else{
                flag_tallyCounterValue_6 = Boolean.FALSE;
            }
        } else{
            flag_tallyCounterValue_6 = Boolean.FALSE;
        }

        FinalResults updatedValues = new FinalResults(
                imageName,
                watermeter,
                pointerValue_1.getValue(),
                pointerValue_2.getValue(),
                pointerValue_3.getValue(),
                flag_pointervalue,
                tallyCounterValue_1.getValue(),
                flag_tallyCounterValue_1,
                tallyCounterValue_2.getValue(),
                flag_tallyCounterValue_2,
                tallyCounterValue_3.getValue(),
                flag_tallyCounterValue_3,
                tallyCounterValue_4.getValue(),
                flag_tallyCounterValue_4,
                tallyCounterValue_5.getValue(),
                flag_tallyCounterValue_5,
                tallyCounterValue_6.getValue(),
                flag_tallyCounterValue_6);
        Call<FinalResults> call = fileService.updateValues(imageName, updatedValues);
        call.enqueue(new Callback<FinalResults>() { // upload the results to the server to be stored into the database
            @Override
            public void onResponse(Call<FinalResults> call, Response<FinalResults> response) {
                if (response.isSuccessful()) {
                    Toast.makeText(ResultsActivity.this, "¡Sus datos han sido almacenados con éxito!", Toast.LENGTH_SHORT).show();
                    openDataStoredActivity(); // When the data are stored successfully the next activity is opened
                }
            }

            @Override
            public void onFailure(Call<FinalResults> call, Throwable t) {
                Toast.makeText(ResultsActivity.this, "ERROR: " + t.getMessage(), Toast.LENGTH_SHORT).show();
            }
        });
    }
}
