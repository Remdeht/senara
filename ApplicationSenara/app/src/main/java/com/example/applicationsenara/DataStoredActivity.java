package com.example.applicationsenara;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

public class DataStoredActivity extends AppCompatActivity {

    private Button returnToMain;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.data_succesfuly_stored);

        returnToMain = (Button) findViewById(R.id.dataStoredReturnToStartButton);
        returnToMain.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openMainActivity();
            }
        });  // Returns to the first screen of  the application
    }

    @Override
    public void onBackPressed() {
        openMainActivity();
    } // When the Back button is pressed the user is directed to the first screen of the application

    public void openMainActivity() {
        Intent intent = new Intent(this, MainActivity.class);
        startActivity(intent);
    }
}
