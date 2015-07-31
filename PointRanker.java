import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import py4j.GatewayServer;
import net.sf.json.JSONObject;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;



public class PointRanker {
    public PointRanker(String name){
        System.out.println("structer " + name);
    }

	public ArrayList<String> json_to_list(String json_line){
		JSONObject json = JSONObject.fromObject(json_line);
		ArrayList<String> src_List = new ArrayList<String>();
		String text = "";
        if (json.containsKey("retweeted_status"))
			text =  json.getJSONObject("retweeted_status").getString("text");
		else
			text =  json.getString("text");
		
		int count=0;
		//Topic
		String pat0 = "#\\w+";
		if (Pattern.compile(pat0).matcher(text).find()){
			src_List.add(0, "True");
		}
		else{
			src_List.add(0, "False");
		}
		
		//urls
		if (json.getJSONObject("entities").get("urls").toString().length()>2){
			src_List.add(1, "True");
		}
		else{
			src_List.add(1, "False");
		}
		
		//user_mentions
		if (json.getJSONObject("entities").get("user_mentions").toString().length()>2){
			src_List.add(2, "True");
		}
		else{
			src_List.add(2, "False");
		}
		
		//word_count
		int word_count = text.split("\\s+").length;
		src_List.add(3, word_count+"");
		
		//capital
		Pattern pattern = Pattern.compile("\\s[A-Z][a-z]+");
		Matcher matcher = pattern.matcher(text);
		while(matcher.find()){
			 count++;
		}
		src_List.add(4, count+"");
		count = 0;
		
		//get_followrs
		if (json.containsKey("retweeted_status")){
			if((count = json.getJSONObject("retweeted_status").getJSONObject("user").getInt("followers_count"))>0){
				src_List.add(5, Math.log10(count)+"");
			}
			else src_List.add(5,"0");
		}
		else {
			if ((count = json.getJSONObject("user").getInt("followers_count"))>0){
				src_List.add(5, Math.log10(count)+"");
			}
			else{
				src_List.add(5, "0");
			}
		}
		count = 0;
		
		//get_statuses
		if (json.containsKey("retweeted_status")){
			if((count = json.getJSONObject("retweeted_status").getJSONObject("user").getInt("statuses_count"))>0){
				src_List.add(6, Math.log10(count)+"");
			}
			else src_List.add(6,"0");
		}
		else {
			if ((count = json.getJSONObject("user").getInt("statuses_count"))>0){
				src_List.add(6, Math.log10(count)+"");
			}
			else{
				src_List.add(6, "0");
			}
		}
		count = 0;
		
		//character_per_word
		src_List.add(7, (text.length()/word_count) + "");
		
		//has_too_many_ellipsis
		String pat8 = "\\.\\.+";
		if (Pattern.compile(pat8).matcher(text).find()){
			src_List.add(8, "True");
		}
		else{
			src_List.add(8, "False");
		}
		
		//get_retweet_count
		if (json.containsKey("retweeted_status")){
			if((count = json.getJSONObject("retweeted_status").getInt("retweet_count"))>0){
				src_List.add(9, count+"");
			}
			else{
				src_List.add(9, "0");
			}
		}
		else{
			src_List.add(9,"0");
		}
		
		//get_retweet_level
		if (count>0){
			count = (count+"").length();
			src_List.add(10, count+"");
		}
		else{
			src_List.add(10, "0");
		}
		
		//has_personal
		String pat11 = "I\\s|my\\s|\\sme";
		if(Pattern.compile(pat11).matcher(text).find()){
			src_List.add(11, "True");
		}
		else{
			src_List.add(11, "False");
		}
		
		return src_List;
	}
	

	public Instance make_instance(String json_line) throws FileNotFoundException, IOException{
		ArrayList<String> features = json_to_list(json_line);
		String dataset_path = "result_test4weka.arff";
		
		Instances dataset = new Instances(new BufferedReader(new FileReader(dataset_path))); 
		dataset.setClassIndex(dataset.numAttributes() - 1); 
		
		FastVector nominal_values_boolean = new FastVector(2); 
		nominal_values_boolean.addElement("True"); 
		nominal_values_boolean.addElement("False");
		
		Attribute Topic = new Attribute("Topic",nominal_values_boolean);//0
		Attribute Url = new Attribute("Url",nominal_values_boolean);//1
		Attribute Mention = new Attribute("Mention",nominal_values_boolean);//2 
		Attribute WordCount = new Attribute("WordCount");//3 
		Attribute Capital = new Attribute("Capital");//4
		Attribute Followers = new Attribute("Followers");//5 
		Attribute Statuses = new Attribute("Statuses");//6 
		Attribute CharacterPerWord = new Attribute("CharacterPerWord");//7 
		Attribute ellipsis = new Attribute("ellipsis",nominal_values_boolean);//8 
		Attribute RetweetCount = new Attribute("RetweetCount");//9 
		Attribute RetweetLevel = new Attribute("RetweetLevel");//10
		Attribute Personal = new Attribute("Personal",nominal_values_boolean);//11
		
		FastVector nominal_values_one_zero = new FastVector(2); 
		nominal_values_one_zero.addElement("1"); 
		nominal_values_one_zero.addElement("0");
		
		Attribute Importance = new Attribute("Importance",nominal_values_one_zero);//12 
		
		Instance inst = new Instance(13); 
		inst.setDataset(dataset);
		
		inst.setValue(0, features.get(0));
		inst.setValue(1, features.get(1));
		inst.setValue(2, features.get(2));
		inst.setValue(3, Double.parseDouble(features.get(3)));
		inst.setValue(4, Double.parseDouble(features.get(4)));
		inst.setValue(5, Double.parseDouble(features.get(5)));
		inst.setValue(6, Double.parseDouble(features.get(6)));
		inst.setValue(7, Double.parseDouble(features.get(7)));
		inst.setValue(8, features.get(8));
		inst.setValue(9, Double.parseDouble(features.get(9)));
		inst.setValue(10, Double.parseDouble(features.get(10)));
		inst.setValue(11, features.get(11));
		inst.setMissing(12);
		
		return inst;
	}
	

	public Classifier load_classifier(String classifier_path) throws Exception{
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(classifier_path));
		Classifier cls = (Classifier) ois.readObject();
		ois.close();
		return cls;
	}
	

	public double predict(String json_line) throws Exception{
		Classifier test_classifier = load_classifier("Trec681_att.model");
		Instance trec_instance = make_instance(json_line);
		double[] clsLabel_double_array = test_classifier.distributionForInstance(trec_instance);
        // System.out.println(clsLabel_double_array[0]);
		return clsLabel_double_array[0];
	}

	
	public static void main(String[] args) throws Exception {
		PointRanker ranker = new PointRanker("test");
        GatewayServer server = new GatewayServer(ranker);
        server.start();
		/*BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("result/day_20/json/profile_MB226.txt")));
		String ss = br.readLine();
        // System.out.println(ss);
		ranker.predict(ss);
		br.close();*/
	}

}
