package com.example.applicationsenara;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.location.Location;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.applicationsenara.model.FileInfo;
import com.example.applicationsenara.model.ImageToProcess;
import com.example.applicationsenara.remote.APIUtils;
import com.example.applicationsenara.remote.FileService;
import com.google.android.gms.location.FusedLocationProviderClient;
import com.google.android.gms.location.LocationServices;
import com.google.android.gms.tasks.OnSuccessListener;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.Locale;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class CameraActivity extends AppCompatActivity {

    private Integer cameraFacing;  // Camera
    private String cameraId;
    private CameraManager cameraManager;
    private CameraDevice cameraDevice;
    private CameraCaptureSession cameraCaptureSession;

    private Size previewSize;  // Image
    private File photo;
    private File galleryFolder;
    private ImageView imageCaptured;
    private TextureView textureView;
    private TextureView.SurfaceTextureListener surfaceTextureListener;

    private Handler backgroundHandler;  // Uploading the Image
    private HandlerThread backgroundThread;
    private FileService fileService;
    private CaptureRequest captureRequest;
    private CaptureRequest.Builder captureRequestBuilder;
    private CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice cameraDevice) {
            CameraActivity.this.cameraDevice = cameraDevice;
            createPreviewSession();
        }

        @Override
        public void onDisconnected(CameraDevice cameraDevice) {
            cameraDevice.close();
            CameraActivity.this.cameraDevice = null;
        }

        @Override
        public void onError(CameraDevice cameraDevice, int error) {
            cameraDevice.close();
            CameraActivity.this.cameraDevice = null;
        }
    };

    private Button confirmButton;  // Buttons
    private Button retakeButton;
    private ImageButton captureButton;
    private int isClicked = 0;  // To track if the capturebutton is clicked

    private FusedLocationProviderClient fusedLocationClient;  // To obtain the user's location
    private double latitude;
    private double longitude;


    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.camera);

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this);
        fileService = APIUtils.getFileService();  // Service used to contact the backend

        textureView = findViewById(R.id.camera_preview);  // Shows the camera Preview
        imageCaptured = (ImageView) findViewById(R.id.photoImageView);  // Shows the photo taken
        captureButton = (ImageButton) findViewById(R.id.button_capture);

        textureView.bringToFront();
        captureButton.bringToFront();

        cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        cameraFacing = CameraCharacteristics.LENS_FACING_BACK;

        surfaceTextureListener = new TextureView.SurfaceTextureListener() {
            @Override
            public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture, int width, int height) {
                setUpCamera();
                openCamera(); // Activates the Camera and shows the camerapreview to the user
            }

            @Override
            public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture, int width, int height) {

            }

            @Override
            public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {
                return false;
            }

            @Override
            public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) {

            }
        };

        captureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                onCaptureButtonClicked();

                confirmButton.setOnClickListener(new View.OnClickListener() {  //

                    @Override
                    public void onClick(View v) {
                        if (photo != null && isClicked == 0) {
                            uploadImageToServer(photo);  // Image is uploaded to the backend when the confirm button is clicked
                            confirmButton.setEnabled(false);
                            confirmButton.bringToFront();
                            retakeButton.setEnabled(false);
                            retakeButton.bringToFront();
                        }
                        isClicked = 1;

                    }
                });

                retakeButton.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) { // Resets to the preview, in order for the user to take another picture
                        captureButton.setEnabled(true);
                        captureButton.setVisibility(View.VISIBLE);
                        textureView.bringToFront();
                        captureButton.bringToFront();
                        confirmButton.setVisibility(View.INVISIBLE);
                        retakeButton.setVisibility(View.INVISIBLE);
                        imageCaptured.setVisibility(View.INVISIBLE);
                        unlock();
                    }
                });
            }
        });
    }

    protected void onCaptureButtonClicked() {

        captureButton.setEnabled(false); // Disable the capture button
        captureButton.setVisibility(View.INVISIBLE);

        confirmButton = findViewById(R.id.savePicture); // Activate the confirm button
        confirmButton.setVisibility(View.VISIBLE);
        confirmButton.bringToFront();
        confirmButton.setEnabled(true);

        retakeButton = findViewById(R.id.retakePicture); // Activate the retake button
        retakeButton.setVisibility(View.VISIBLE);
        retakeButton.bringToFront();
        retakeButton.setEnabled(true);

        imageCaptured.setVisibility(View.VISIBLE);  // Makes the imageview visible to the user
        imageCaptured.bringToFront();

        if (ContextCompat.checkSelfPermission(CameraActivity.this,
                Manifest.permission.ACCESS_FINE_LOCATION)
                == PackageManager.PERMISSION_GRANTED) {
            fusedLocationClient.getLastLocation()
                    .addOnSuccessListener(CameraActivity.this, new OnSuccessListener<Location>() {
                        @Override
                        public void onSuccess(Location location) {
                            if (location != null) {
                                longitude = location.getLongitude();
                            }
                            if (location != null) {
                                latitude = location.getLatitude();
                            }
                        }
                    }); // Gets the user's location to determine the position of the nearest water meter
            lock();  // Locks the camerapreview
            createImageGallery();
            FileOutputStream outputPhoto = null;
            try {
                // Creates an imagefile from the camerapreview. The ImageView imageCaptured shows this image to the user
                photo = createImageFile(galleryFolder);
                outputPhoto = new FileOutputStream(photo);
                textureView.getBitmap()
                        .compress(Bitmap.CompressFormat.PNG, 100, outputPhoto);
                imageCaptured.setImageBitmap(BitmapFactory.decodeFile(photo.getAbsolutePath()));

            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                try {
                    if (outputPhoto != null) {
                        outputPhoto.close();
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        openBackgroundThread();
        if (textureView.isAvailable()) {
            setUpCamera();
            openCamera();
        } else {
            textureView.setSurfaceTextureListener(surfaceTextureListener);
        }
    }

    private void setUpCamera() {
        // Sets up the camera, making sure the camera on the back of the  device is used
        try {
            for (String cameraId : cameraManager.getCameraIdList()) {
                CameraCharacteristics cameraCharacteristics = cameraManager.getCameraCharacteristics(cameraId);
                if (cameraCharacteristics.get(CameraCharacteristics.LENS_FACING) ==
                        cameraFacing) {
                    StreamConfigurationMap streamConfigurationMap = cameraCharacteristics.get(
                            CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                    previewSize = streamConfigurationMap.getOutputSizes(SurfaceTexture.class)[0];
                    this.cameraId = cameraId;
                }
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void openCamera() {
        // Activates the camera
        try {
            if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED) {
                cameraManager.openCamera(cameraId, stateCallback, backgroundHandler);
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void openBackgroundThread() {
        backgroundThread = new HandlerThread("camera_background_thread");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
    }

    @Override
    protected void onStop() {
        super.onStop();
        closeCamera();
        closeBackgroundThread();
    }

    private void closeCamera() {
        // Closes the Camera
        if (cameraCaptureSession != null) {
            cameraCaptureSession.close();
            cameraCaptureSession = null;
        }

        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
        }
    }

    private void closeBackgroundThread() {
        if (backgroundHandler != null) {
            backgroundThread.quitSafely();
            backgroundThread = null;
            backgroundHandler = null;
        }
    }

    private void lock() {
        try {
            cameraCaptureSession.capture(captureRequestBuilder.build(),
                    null, backgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void unlock() {
        try {
            cameraCaptureSession.setRepeatingRequest(captureRequestBuilder.build(),
                    null, backgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void createPreviewSession() {
        // starts the camera preview
        try {
            SurfaceTexture surfaceTexture = textureView.getSurfaceTexture();
            surfaceTexture.setDefaultBufferSize(previewSize.getWidth(), previewSize.getHeight());
            Surface previewSurface = new Surface(surfaceTexture);
            captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(previewSurface);

            cameraDevice.createCaptureSession(Collections.singletonList(previewSurface),
                    new CameraCaptureSession.StateCallback() {

                        @Override
                        public void onConfigured(CameraCaptureSession cameraCaptureSession) {
                            if (cameraDevice == null) {
                                return;
                            }

                            try {
                                captureRequest = captureRequestBuilder.build();
                                CameraActivity.this.cameraCaptureSession = cameraCaptureSession;
                                CameraActivity.this.cameraCaptureSession.setRepeatingRequest(captureRequest,
                                        null, backgroundHandler);
                            } catch (CameraAccessException e) {
                                e.printStackTrace();
                            }
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession cameraCaptureSession) {

                        }
                    }, backgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void uploadImageToServer(final File imageToUpload) {

        RequestBody requestBody = RequestBody.create(MediaType.parse("multipart/form-data"), imageToUpload);
        MultipartBody.Part body = MultipartBody.Part.createFormData("image", imageToUpload.getName(), requestBody);

        Call<FileInfo> call = fileService.upload(body);
        call.enqueue(new Callback<FileInfo>() {
            @Override
            public void onResponse(Call<FileInfo> call, Response<FileInfo> response) {
                if (response.isSuccessful()) {
                    Toast.makeText(CameraActivity.this, "La imagen se cargó correctamente en el servidor.", Toast.LENGTH_SHORT).show();
                    activateProcessing(imageToUpload.getName(), latitude, longitude);
                }
                else {
                    isClicked = 0;
                    confirmButton.setEnabled(false);
                    confirmButton.setVisibility(View.INVISIBLE);
                    retakeButton.setEnabled(false);
                    retakeButton.setVisibility(View.INVISIBLE);
                    captureButton.setEnabled(true);
                    captureButton.setVisibility(View.VISIBLE);
                    imageCaptured.setVisibility(View.INVISIBLE);
                    textureView.bringToFront();
                    captureButton.bringToFront();
                }
            }

            @Override
            public void onFailure(Call<FileInfo> call, Throwable t) {
                Toast.makeText(CameraActivity.this, "ERROR: " + t.getMessage(), Toast.LENGTH_SHORT).show();
                isClicked = 0;
                confirmButton.setEnabled(false);
                confirmButton.setVisibility(View.INVISIBLE);
                retakeButton.setEnabled(false);
                retakeButton.setVisibility(View.INVISIBLE);
                captureButton.setEnabled(true);
                captureButton.setVisibility(View.VISIBLE);
                imageCaptured.setVisibility(View.INVISIBLE);
                textureView.bringToFront();
                captureButton.bringToFront();
            }
        });

    }

    private void activateProcessing(final String imageLocation, final double latitude, final double longitude) {
        // Activates the image processing on the server, and opens the next activity when completed with success
        Call<ImageToProcess> processing_call = fileService.process(imageLocation, latitude, longitude);
        processing_call.enqueue(new Callback<ImageToProcess>() {
            @Override
            public void onResponse(Call<ImageToProcess> call, Response<ImageToProcess> response) {
                if (response.isSuccessful()) {
                    ImageToProcess results = response.body();

                    if (results.isSafely_executed()) {
                        Toast.makeText(CameraActivity.this, "El procesamiento de la imagen fue exitoso", Toast.LENGTH_SHORT).show();
                        closeCamera();

                        openMapActivity(
                                imageLocation,
                                results.getWatermeter(),
                                results.getLatitude(),
                                results.getLongitude(),
                                results.getTally_counter_1(),
                                results.getTally_counter_2(),
                                results.getTally_counter_3(),
                                results.getTally_counter_4(),
                                results.getTally_counter_5(),
                                results.getTally_counter_6(),
                                results.getPointer_value_1(),
                                results.getPointer_value_2(),
                                results.getPointer_value_3()
                        );
                        photo.delete(); // Deletes the image and app folder from the device
                        galleryFolder.delete();
                        finish();
                    } else {
                        Toast.makeText(CameraActivity.this, "El procesamiento de la imagen no se pudo completar. Por favor, inténtalo de nuevo.", Toast.LENGTH_SHORT).show();
                        isClicked = 0;
                        confirmButton.setEnabled(false);
                        confirmButton.setVisibility(View.INVISIBLE);
                        retakeButton.setEnabled(false);
                        retakeButton.setVisibility(View.INVISIBLE);
                        captureButton.setEnabled(true);
                        captureButton.setVisibility(View.VISIBLE);
                        imageCaptured.setVisibility(View.INVISIBLE);
                        textureView.bringToFront();
                        captureButton.bringToFront();
                    }
                }
            }

            @Override
            public void onFailure(Call<ImageToProcess> call, Throwable t) {
                Toast.makeText(CameraActivity.this, "ERROR: " + t.getMessage(), Toast.LENGTH_SHORT).show();

            }
        });
    }

    private void openMapActivity(String name, String watermeter,
                                 double latitude, double longitude, ArrayList<Integer> tc_1,
                                 ArrayList<Integer> tc_2, ArrayList<Integer> tc_3, ArrayList<Integer> tc_4, ArrayList<Integer> tc_5,
                                 ArrayList<Integer> tc_6, int pointer_val_1, int pointer_val_2, int pointer_val_3) {
        Intent intent = new Intent(this, MapActivity.class);
        Bundle b = new Bundle();
        b.putString("name", name);
        b.putString("watermeter", watermeter);
        b.putDouble("latitude", longitude); // TODO - FIX the naming
        b.putDouble("longitude", latitude);
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

    private void createImageGallery() {
        File storageDirectory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
        galleryFolder = new File(storageDirectory, getResources().getString(R.string.app_name));
        if (!galleryFolder.exists()) {
            boolean wasCreated = galleryFolder.mkdirs();
            if (!wasCreated) {
                Log.e("CapturedImages", "Failed to create directory");
            }
        }
    }

    private File createImageFile(File galleryFolder) throws IOException {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        String imageFileName = "image_" + timeStamp + "_";
        return File.createTempFile(imageFileName, ".jpg", galleryFolder);
    }

}
