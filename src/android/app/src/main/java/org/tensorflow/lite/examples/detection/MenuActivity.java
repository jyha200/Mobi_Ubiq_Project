package org.tensorflow.lite.examples.detection;

import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.os.Bundle;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ListView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.tensorflow.lite.examples.detection.env.Logger;

public class MenuActivity extends AppCompatActivity {
  private Button returnButton;

  private static final Logger LOGGER = new Logger();
  static public HashMap<String, String> reference_guide = new HashMap<String, String>(){
    {
      put("machine", "Can use washing machine");
      put("nowater", "Do not wash with water");
      put("bleach_O", "Can use bleach");
      put("bleach_X", "Do not use bleach");
      put("dryer_O", "Can use dry machine");
      put("dryer_X", "Do not use dry mach");
      put("wring_O", "Wring gently by han");
      put("wring_X", "Do not wring");
      put("sun", "Dry under sunlight");
      put("shade", "Dry under shade");
      put("iron_O", "Can iron");
      put("iron_X", "Do not iron");
      put("dryclean_O", "Dry clean");
      put("dryclean_X", "No dry clean");
    }
  };

  @Override
  protected void onCreate(@Nullable Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.tfe_od_activity_menu);
    String[] LIST_MENU = getClothIdxs();

    ArrayAdapter adapter = new ArrayAdapter(this, android.R.layout.simple_list_item_1, LIST_MENU);

    ListView listview = (ListView) findViewById(R.id.cloth_list);
    listview.setAdapter(adapter);
    listview.setOnItemClickListener(new AdapterView.OnItemClickListener() {
      @Override
      public void onItemClick(AdapterView parent, View v, int position, long id) {
        // get TextView's Text.
        String strText = (String) parent.getItemAtPosition(position);
        AlertDialog.Builder builder = new AlertDialog.Builder(MenuActivity.this);
        builder.setTitle(strText);
        String[] labels = getCareLabels(strText);
        List<String> msg_list = new ArrayList<>();
        for (int i = 0 ; i < labels.length ; i++) {
          String guide = MenuActivity.reference_guide.get(labels[i]);
          msg_list.add(guide);
        }

        CharSequence[] items = msg_list.toArray(new String[msg_list.size()]);
        builder.setItems(items, new DialogInterface.OnClickListener() {
          @Override
          public void onClick(DialogInterface dialogInterface, int i) {
            String text = items[i].toString();
            Toast.makeText(getApplicationContext(), text, Toast.LENGTH_SHORT).show();
          }
        });
        builder.setNeutralButton("Confirm", null);
        builder.create().show();
      }
    });
  }

  String[] getClothIdxs() {
    String[] ret = {"pink T-shirt", "black jean", "baseball jumper"};
    return ret;
  }

  String[] getCareLabels(String cloth_idx) {
    String[] ret = {"machine", "bleach_O", "dryer_X", "wring_O","sun","iron_X"} ;
    return ret;
  }
}