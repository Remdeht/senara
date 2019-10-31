package com.example.applicationsenara;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import com.example.applicationsenara.model.NearbyWatermeters;
import com.example.applicationsenara.remote.APIUtils;
import com.example.applicationsenara.remote.FileService;
import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.OnMapReadyCallback;
import com.google.android.gms.maps.SupportMapFragment;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.Marker;
import com.google.android.gms.maps.model.MarkerOptions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MapActivity extends AppCompatActivity {

    private Bundle b;  // Bundle b will contain the information passed on from the previous activity
    private FileService fileService;

    private SupportMapFragment mapFragment;
    private GoogleMap mMap;
    private double latitude;
    private double longitude;
    private static final float ZOOM = 16;  // Zoomlevel on the map
    private String selectedWatermeter;  // The nearest water meter based on the user's location

    // Layout

    private Button yesButton;
    private Button noButton;
    private Button confirmButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.map);
        fileService = APIUtils.getFileService();

        b = getIntent().getExtras(); // Gets the info from the Intent

        if (b != null) {
            selectedWatermeter = b.getString("watermeter");  // Get the nearest water meter info
            latitude = b.getDouble("latitude");
            longitude = b.getDouble("longitude");
        }

        // Portrays an interactive google map
        mapFragment = (SupportMapFragment) getSupportFragmentManager().findFragmentById(R.id.map);
        mapFragment.getMapAsync(new OnMapReadyCallback() {
            @Override
            public void onMapReady(GoogleMap googleMap) {
                mMap = googleMap;
                mMap.getUiSettings().setAllGesturesEnabled(true);
                moveCamera(new LatLng(latitude, longitude), ZOOM, selectedWatermeter);

                if (ContextCompat.checkSelfPermission(MapActivity.this,
                        Manifest.permission.ACCESS_FINE_LOCATION)
                        == PackageManager.PERMISSION_GRANTED) {
                    mMap.setMyLocationEnabled(true);
                }

                // Buttons

                yesButton = findViewById(R.id.yes_button_map_confirm);
                noButton = findViewById(R.id.no_button_map_confirm);
                yesButton.bringToFront();
                noButton.bringToFront();
                confirmButton = findViewById(R.id.button_map_confirmation);
                confirmButton.setVisibility(View.INVISIBLE);
                confirmButton.setEnabled(false);

                yesButton.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        openResultsActivity(
                                b.getString("name"),
                                selectedWatermeter,
                                b.getIntegerArrayList("tc_1"),
                                b.getIntegerArrayList("tc_2"),
                                b.getIntegerArrayList("tc_3"),
                                b.getIntegerArrayList("tc_4"),
                                b.getIntegerArrayList("tc_5"),
                                b.getIntegerArrayList("tc_6"),
                                b.getInt("pointer_val_1"),
                                b.getInt("pointer_val_2"),
                                b.getInt("pointer_val_3")
                        );
                        finish();
                    }
                });

                noButton.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        getNearbyWatermeters();
                        noButton.setEnabled(false);
                        noButton.setVisibility(View.INVISIBLE);
                        yesButton.setEnabled(false);
                        yesButton.setVisibility(View.INVISIBLE);
                    }
                });

                confirmButton.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        openResultsActivity(
                                b.getString("name"),
                                selectedWatermeter,
                                b.getIntegerArrayList("tc_1"),
                                b.getIntegerArrayList("tc_2"),
                                b.getIntegerArrayList("tc_3"),
                                b.getIntegerArrayList("tc_4"),
                                b.getIntegerArrayList("tc_5"),
                                b.getIntegerArrayList("tc_6"),
                                b.getInt("pointer_val_1"),
                                b.getInt("pointer_val_2"),
                                b.getInt("pointer_val_3")
                        );
                        finish();
                    }
                });

            }
        });
    }

    private void moveCamera(LatLng latLng, float zoom, String title) {
        // Moves the Camera to the location of the nearest Water Meter and adds a Marker
        mMap.moveCamera(CameraUpdateFactory.newLatLngZoom(latLng, zoom));
        MarkerOptions marker = new MarkerOptions()
                .position(latLng)
                .title(title);
        mMap.addMarker(marker);

        mMap.setOnMarkerClickListener(new GoogleMap.OnMarkerClickListener() {
            @Override
            public boolean onMarkerClick(Marker marker) {
                selectedWatermeter = marker.getTitle();
                marker.showInfoWindow();
                return true;
            }
        });


    }

    private void openResultsActivity(String name, String watermeter, ArrayList<Integer> tc_1, ArrayList<Integer> tc_2,
                                     ArrayList<Integer> tc_3, ArrayList<Integer> tc_4, ArrayList<Integer> tc_5, ArrayList<Integer> tc_6,
                                     int pointer_val_1, int pointer_val_2, int pointer_val_3) {
        Intent intent = new Intent(this, ResultsActivity.class);
        Bundle b = new Bundle();
        b.putString("name", name);
        b.putString("watermeter", watermeter);
        b.putIntegerArrayList("tc_1", tc_1);
        b.putIntegerArrayList("tc_2", tc_2);
        b.putIntegerArrayList("tc_3", tc_3);
        b.putIntegerArrayList("tc_4", tc_4);
        b.putIntegerArrayList("tc_5", tc_5);
        b.putIntegerArrayList("tc_6", tc_6);
        b.putInt("pointer_val_1", pointer_val_1);
        b.putInt("pointer_val_2", pointer_val_2);
        b.putInt("pointer_val_3", pointer_val_3);
        intent.putExtras(b);
        startActivity(intent);
    }

    private void getNearbyWatermeters() {
        // Activated when the user wants to change the water meter picked based on his location.
        // Gets the nearest 5 water meters from the server
        Call<NearbyWatermeters> processing_call = fileService.get_nearby_watermeters(latitude, longitude);
        processing_call.enqueue(new Callback<NearbyWatermeters>() {
            @Override
            public void onResponse(Call<NearbyWatermeters> call, Response<NearbyWatermeters> response) {
                if (response.isSuccessful()) {
                    NearbyWatermeters result = response.body();
                    addNearbyWatermeterMarkers(result);
                } else {
                    Toast.makeText(MapActivity.this, "No es posible contactar al servidor", Toast.LENGTH_SHORT);
                }
            }

            @Override
            public void onFailure(Call<NearbyWatermeters> call, Throwable t) {
                Toast.makeText(MapActivity.this, "No es posible contactar al servidor", Toast.LENGTH_SHORT);
            }

        });
    }

    private void addNearbyWatermeterMarkers(NearbyWatermeters result) {
        // Adds marker for each of the 5 nearest water meters

        MarkerOptions watermeter_1 = createMarkerObject(result.getWatermeter_1());
        MarkerOptions watermeter_2 = createMarkerObject(result.getWatermeter_2());
        MarkerOptions watermeter_3 = createMarkerObject(result.getWatermeter_3());
        MarkerOptions watermeter_4 = createMarkerObject(result.getWatermeter_4());
        MarkerOptions watermeter_5 = createMarkerObject(result.getWatermeter_5());

        List<MarkerOptions> markers = Arrays.asList(
                watermeter_1, watermeter_2, watermeter_3, watermeter_4, watermeter_5);

        for (final MarkerOptions marker : markers) {
            if (marker != null){
                mMap.addMarker(marker);

            }
        }

        mMap.setOnMarkerClickListener(new GoogleMap.OnMarkerClickListener() {
            @Override
            public boolean onMarkerClick(Marker marker) {
                selectedWatermeter = marker.getTitle();
                marker.showInfoWindow();
                confirmButton.setEnabled(true);
                confirmButton.setVisibility(View.VISIBLE);

                return true;
            }
        });
    }



    private MarkerOptions createMarkerObject(ArrayList<String> apiResult) {
        if (apiResult.get(0) != null) {
            Float lng = Float.parseFloat(apiResult.get(1));
            Float lat = Float.parseFloat(apiResult.get(2));
            LatLng latLng = new LatLng(lat, lng);
            MarkerOptions marker = new MarkerOptions()
                    .position(latLng)
                    .visible(true)
                    .title(apiResult.get(0));
            return marker;
        } else {
            return null;
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

}


