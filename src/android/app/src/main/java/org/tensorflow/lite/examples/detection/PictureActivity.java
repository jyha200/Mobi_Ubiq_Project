package org.tensorflow.lite.examples.detection;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckedTextView;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.examples.detection.env.Logger;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class PictureActivity extends AppCompatActivity {
  private Button returnButton;
  private static final Logger LOGGER = new Logger();

  static public HashMap<String, String> reference_guide = new HashMap<String, String>(){
    {
      put("machine", "Can use washing machine");
      put("handwash", "Can only do hand wash");
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
  static public HashMap<String, Integer> reference_images = new HashMap<String, Integer>(){
    {
      put("machine", R.drawable.machine);
      put("handwash", R.drawable.handwash);
      put("nowater", R.drawable.nowater);
      put("bleach_O", R.drawable.bleach_o);
      put("bleach_X", R.drawable.bleach_x);
      put("dryer_O", R.drawable.dryer_o);
      put("dryer_X", R.drawable.dryer_x);
      put("wring_O", R.drawable.wring_o);
      put("wring_X", R.drawable.wring_x);
      put("sun", R.drawable.sun);
      put("shade", R.drawable.shade);
      put("iron_O", R.drawable.iron_o);
      put("iron_X", R.drawable.iron_x);
      put("dryclean_O", R.drawable.dryclean_o);
      put("dryclean_X", R.drawable.dryclean_x);
    }
  };
  protected class LegendListAdapter extends ArrayAdapter<String> {
    private Context context;
    private ViewHolder holder;
    private String[] guides;
    // 자료를 저장할 클래스를 만든다.
    class ViewHolder {
      int position;
      ImageView icon;
      TextView title;
    };

    public LegendListAdapter(Context context, int textViewResourceId, List<String> objects) {
      super(context, textViewResourceId, objects);
      this.guides = objects.toArray(new String[objects.size()]);
      // TODO Auto-generated constructor stub
      this.context = context;
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
      View row = convertView;

      // 선택항목 하나에 대한 view의 레이아웃 설정
      if(row == null) {
        LayoutInflater inflator = (LayoutInflater)context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        row = inflator.inflate(R.layout.tfe_od_select_legend_item, null);
      }

      // 해당항목의 정보를 Holder를 만들어서 저장한다.
      holder = new ViewHolder();
      // 레이아웃에서 출력할 위젯을 찾아낸다.
      holder.icon = (ImageView)row.findViewById(R.id.ivItemLgndIcon);
      holder.title = (TextView)row.findViewById(R.id.ctvItemLgndTitle);
      holder.position = position;

      if(position<guides.length) {
        // 표시값 세팅
        holder.icon.setImageDrawable(getResources().getDrawable(reference_images.get(guides[position])));
        holder.title.setText(reference_guide.get(guides[position]));
      }

      // holder를 저장해놓는다.
      row.setTag(holder);

      return(row);
    }
  }

  @Override
  protected void onCreate(@Nullable Bundle savedInstanceState) {
    super.onCreate(null);
    setContentView(R.layout.tfe_od_activity_picture);
    ImageView imageView = findViewById(R.id.picture_image_view);
    imageView.setImageResource(R.drawable.sample_image);
    List<String> labels = getLabels();
    AlertDialog.Builder builder = new AlertDialog.Builder(PictureActivity.this);
    builder.setTitle("Confirm Recognized Labels");

    if (labels.size() > 0) {
      LegendListAdapter adapter = new LegendListAdapter(PictureActivity.this, R.layout.tfe_od_select_legend_item, labels);
      builder.setAdapter(adapter, null);
      builder.setPositiveButton("Confirm", new DialogInterface.OnClickListener() {
        @Override
        public void onClick(DialogInterface dialog, int id) {
          AlertDialog.Builder builder = new AlertDialog.Builder(PictureActivity.this);
          builder.setTitle("Insert name of this cloth");
          EditText input = new EditText(PictureActivity.this);
          builder.setView(input);
          builder.setPositiveButton("Done", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int id) {
              String value = input.getText().toString();
            }
          });
          builder.create().show();
        }
      });
      builder.setNegativeButton("Cancel", null);
    } else {
      builder.setMessage("No label recognized");
      builder.setNeutralButton("OK", null);
    }
    builder.create().show();
  }

  List<String> getLabels() {
    List<String> ret = new ArrayList<>();
    ret.add("machine");
    ret.add("bleach_O");
    return ret;
  }
}