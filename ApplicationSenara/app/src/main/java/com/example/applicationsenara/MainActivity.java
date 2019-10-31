package com.example.applicationsenara;

import android.app.Dialog;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageButton;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.gms.common.ConnectionResult;
import com.google.android.gms.common.GoogleApiAvailability;
import com.karan.churi.PermissionManager.PermissionManager;

public class MainActivity extends AppCompatActivity{

    private static final int ERROR_DIALOG_REQUEST = 9001;
    private int isClicked = 0;
    PermissionManager permissionManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if(isServicesOK()) {
            permissionManager = new PermissionManager() {};
            permissionManager.checkAndRequestPermissions(this); // Requests permissions if needed

            setContentView(R.layout.activity_main);

            ImageButton cameraButton = findViewById(R.id.cameraButton);
            cameraButton.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    openCameraActivity();
                }
            });

            ImageButton infoButton = findViewById(R.id.InfoButton);
            infoButton.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    if (isClicked == 0) {
                        openInfoActivity();
                    }
                    isClicked = 1;
                }


            });
        }
    }

    @Override
        public void onRequestPermissionsResult(int requestCode,String permissions[], int[] grantResults) {
            permissionManager.checkResult(requestCode,permissions, grantResults);  // Makes sure the application has all the needed permissions
        }
        
    public void openCameraActivity() {
        Intent intent = new Intent(this, CameraActivity.class);
        startActivity(intent);
    }

    public void openInfoActivity() {
        Intent intent = new Intent(this, OrgInfoActivity.class);
        startActivity(intent);
    }

    public boolean isServicesOK(){  // Checks the Google Maps API
        int available = GoogleApiAvailability.getInstance().isGooglePlayServicesAvailable(MainActivity.this);

        if(available == ConnectionResult.SUCCESS){
            return true; //everything is fine and the user can make map requests
        }
        else if(GoogleApiAvailability.getInstance().isUserResolvableError(available)){
            Dialog dialog = GoogleApiAvailability.getInstance().getErrorDialog(MainActivity.this, available, ERROR_DIALOG_REQUEST);
            dialog.show(); //an error occured but its resolvable
        }else{
            Toast.makeText(this, "Problemas técnicos, por favor intente nuevamente más tarde", Toast.LENGTH_SHORT).show();
        }
        return false;
    }
}
